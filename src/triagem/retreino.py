"""Etapas do retreino orquestrado, chamadas pela DAG do Airflow.

A logica fica aqui (e nao na DAG) para ser testavel sem o Airflow. Fluxo seguro:
treina em um diretorio de staging, converte para ONNX (com portao de paridade), valida o
resultado contra metricas minimas e so entao publica no diretorio que a API le. Um treino
ruim ou interrompido nunca substitui o modelo em uso.
"""

import json
import logging
import os
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from triagem.classificador import Classificador
from triagem.config import Backend, Settings
from triagem.corpus import Corpus, carregar_corpus
from triagem.dados import ARQUIVOS, baixar_dataset
from triagem.treino import FOLDS_PADRAO, RECALL_ALVO_PADRAO, executar_treino

logger = logging.getLogger(__name__)

# Portao de qualidade. O modelo atual tem F1 de urgencia 0,63 e recall 0,82 no teste; os
# minimos deixam folga para a variacao numerica entre maquinas (D-020) e ainda barram um
# treino quebrado ou muito pior.
F1_URGENCIA_MINIMO = 0.55
RECALL_URGENTE_MINIMO = 0.75

PASTA_STAGING = ".staging"
ARQUIVO_TREINO = "medical_tc_train.csv"
ARQUIVO_TESTE = "medical_tc_test.csv"
LAUDO_DE_FUMACA = "Patient with acute myocardial infarction and chest pain"


class ModeloReprovadoError(RuntimeError):
    """O modelo treinado nao atingiu as metricas minimas e nao deve ser publicado."""


def carregar_dados(settings: Settings) -> tuple[Corpus, Corpus]:
    """Baixa (se preciso) e valida o dataset, e le os CSVs de treino e teste.

    O download confere o SHA-256 de cada arquivo; o CSV e validado por `carregar_corpus`.
    """
    caminhos = baixar_dataset(settings)
    treino = carregar_corpus(caminhos[ARQUIVO_TREINO])
    teste = carregar_corpus(caminhos[ARQUIVO_TESTE])
    logger.info("Dados prontos: %d documentos de treino e %d de teste", len(treino), len(teste))
    return treino, teste


def treinar_em_staging(
    settings: Settings,
    treino: Corpus,
    teste: Corpus,
    folds: int = FOLDS_PADRAO,
    recall_alvo: float = RECALL_ALVO_PADRAO,
) -> Path:
    """Treina e grava modelo e metadados em um staging novo, sem tocar no modelo publicado.

    O staging fica dentro de `artifacts_dir`, no mesmo sistema de arquivos, para a publicacao
    ser uma renomeacao atomica.

    Returns:
        Diretorio do staging com `modelo.joblib` e `metadata.json`.
    """
    momento = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    staging = settings.artifacts_dir / PASTA_STAGING / momento
    staging.mkdir(parents=True)
    executar_treino(
        replace(settings, artifacts_dir=staging),
        treino,
        teste,
        folds=folds,
        recall_alvo=recall_alvo,
        hash_treino=ARQUIVOS[ARQUIVO_TREINO],
    )
    return staging


def exportar_onnx_em_staging(settings: Settings, staging: Path, teste: Corpus) -> dict[str, Any]:
    """Converte o modelo do staging para ONNX, com o portao de paridade, ainda no staging.

    Raises:
        ConversaoOnnxError, ParidadeOnnxError: o ONNX nao reproduz o modelo original; nada e
            gravado e o modelo em uso permanece intacto.
    """
    from triagem.onnx_export import exportar_onnx  # so o Airflow e o desenvolvimento tem skl2onnx

    return exportar_onnx(replace(settings, artifacts_dir=staging), teste.textos)


def validar_staging(
    settings: Settings,
    staging: Path,
    f1_minimo: float = F1_URGENCIA_MINIMO,
    recall_minimo: float = RECALL_URGENTE_MINIMO,
    exigir_onnx: bool = True,
) -> dict[str, Any]:
    """Confere se o staging carrega, classifica e atinge as metricas minimas no teste.

    Com `exigir_onnx`, o modelo ONNX tambem precisa existir, pertencer a este treino, carregar e
    classificar: a API usa ONNX por padrao no compose, entao publicar so o scikit-learn
    deixaria a API sem modelo valido.

    Raises:
        ModeloIndisponivelError: artefato ausente, ilegivel ou com versao incompativel (ou
            ONNX ausente/defasado, se exigido).
        ModeloReprovadoError: metricas de teste abaixo dos minimos.
    """
    settings_staging = replace(settings, artifacts_dir=staging, backend=Backend.SKLEARN)
    classificador = Classificador.carregar(settings_staging)
    resultado = classificador.classificar(LAUDO_DE_FUMACA)
    if abs(sum(resultado.probabilidades_urgencia.values()) - 1.0) > 1e-6:
        raise ModeloReprovadoError("As probabilidades de urgencia nao somam 1.")
    versao_onnx = None
    if exigir_onnx:
        classificador_onnx = Classificador.carregar(replace(settings_staging, backend=Backend.ONNX))
        resultado_onnx = classificador_onnx.classificar(LAUDO_DE_FUMACA)
        if resultado_onnx.urgencia != resultado.urgencia:
            raise ModeloReprovadoError(
                "O ONNX e o scikit-learn decidem diferente no laudo de teste."
            )
        versao_onnx = classificador_onnx.versao

    metricas = _ler_metricas_de_teste(settings_staging)
    f1, recall = metricas["f1_macro_urgencia"], metricas["recall_urgente"]
    if f1 < f1_minimo or recall < recall_minimo:
        raise ModeloReprovadoError(
            f"Modelo reprovado: F1 de urgencia {f1:.3f} (minimo {f1_minimo}) e "
            f"recall urgente {recall:.3f} (minimo {recall_minimo})."
        )
    logger.info("Modelo aprovado: F1 urgencia %.3f, recall urgente %.3f", f1, recall)
    return {
        "versao": classificador.versao,
        "versao_onnx": versao_onnx,
        "f1_urgencia": f1,
        "recall_urgente": recall,
    }


def promover_staging(settings: Settings, staging: Path) -> None:
    """Publica o staging validado em `artifacts_dir` e remove o staging.

    Cada arquivo e trocado por renomeacao atomica (o leitor ve o arquivo antigo ou o novo,
    nunca um parcial). O metadado e trocado por ultimo. Como sao tres arquivos, existe uma
    janela de milissegundos entre as trocas; a API so le o modelo na inicializacao (D-023), e
    em modo ONNX recusa subir se o ONNX nao bater com os metadados, entao o risco se limita a
    um restart exatamente nesse instante (e falha de forma explicita, nao serve modelo errado).
    Um staging sem ONNX remove o ONNX publicado antes, para nao sobrar um arquivo de outro treino.

    Raises:
        ValueError: `staging` nao e um staging deste `artifacts_dir` (protege contra mover
            arquivos de outro lugar).
    """
    raiz = (settings.artifacts_dir / PASTA_STAGING).resolve()
    if staging.resolve().parent != raiz:
        raise ValueError(f"{staging} nao e um staging valido de {settings.artifacts_dir}.")
    origem = replace(settings, artifacts_dir=staging)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Ordem: ONNX, modelo e, por ultimo, os metadados (que referenciam o ONNX pelo SHA-256).
    if origem.onnx_path.exists():
        os.replace(origem.onnx_path, settings.onnx_path)
    else:
        settings.onnx_path.unlink(missing_ok=True)  # nunca deixar um ONNX de outro treino
    os.replace(origem.model_path, settings.model_path)
    os.replace(origem.metadata_path, settings.metadata_path)
    shutil.rmtree(staging)
    logger.info("Modelo publicado em %s", settings.artifacts_dir)


def _ler_metricas_de_teste(settings: Settings) -> dict[str, float]:
    metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    return metadata["teste_com_limiar"]
