"""Exportacao do modelo treinado para ONNX, com portao de paridade.

O ONNX Runtime classifica um laudo em ~0,6 ms contra ~5,6 ms do pipeline do scikit-learn,
mas so serve se decidir igual. Por isso a exportacao so grava o arquivo depois de comparar,
no conjunto de teste, as probabilidades e as decisoes de urgencia dos dois motores. O grafo
tem uma correcao conhecida (abaixo) e o portao existe para falhar alto se ela deixar de
bastar, por exemplo apos atualizar o skl2onnx.

Uso: python -m triagem.onnx_export   (depois de `python -m triagem.treino`)

Este modulo usa skl2onnx e onnx, que so existem no ambiente de desenvolvimento e na imagem do
Airflow; a imagem da API so precisa do `triagem.modelo_onnx`.
"""

import argparse
import copy
import hashlib
import json
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime
import skl2onnx
import sklearn
from onnx import TensorProto, helper, numpy_helper
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from triagem.config import Settings
from triagem.corpus import carregar_corpus
from triagem.dados import baixar_dataset
from triagem.modelo_onnx import ENTRADA, PALAVRA_RESERVADA, ModeloOnnx
from triagem.texto import normalizar
from triagem.urgencia import agregar_probabilidades, decidir_urgencia

logger = logging.getLogger(__name__)

OPSET = {"": 17, "ai.onnx.ml": 3}
# Resolucao (em probabilidade) dos calibradores densificados: ver `_densificar_calibrador`.
PASSO_CALIBRACAO = 0.001
# Tolerancias do portao de paridade (escolhas minhas, ver D-033): a diferenca de probabilidade
# cobre a aritmetica de 32 bits do ONNX e a aproximacao da calibracao; a divergencia de
# urgencia e a fracao de laudos cuja decisao muda.
DIFERENCA_MAXIMA_PROBABILIDADE = 0.02
DIVERGENCIA_MAXIMA_URGENCIA = 0.001
# Sufixos acrescentados a laudos reais do teste para exercitar tokens nao-ASCII.
SUFIXO_NAO_ASCII = " resume naive cafe"  # versao acentuada montada abaixo
_NAO_ASCII = " résumé naïve café"


class ConversaoOnnxError(RuntimeError):
    """O modelo nao pode ser convertido para ONNX com seguranca."""


class ParidadeOnnxError(RuntimeError):
    """O modelo ONNX decide de forma diferente do modelo original."""


def converter(pipeline: Pipeline) -> onnx.ModelProto:
    """Converte o pipeline (TF-IDF + classificador com `predict_proba`) para ONNX.

    Raises:
        ConversaoOnnxError: pipeline fora do formato esperado ou token reservado no vocabulario.
    """
    try:
        vetorizador = pipeline.named_steps["tfidf"]
    except KeyError:
        raise ConversaoOnnxError("O pipeline nao tem a etapa 'tfidf'.") from None
    if PALAVRA_RESERVADA in vetorizador.vocabulary_:
        raise ConversaoOnnxError(f"O vocabulario contem o token reservado {PALAVRA_RESERVADA!r}.")
    pipeline = _com_calibracao_densa(pipeline)
    classificador = pipeline.steps[-1][1]
    modelo = convert_sklearn(
        pipeline,
        initial_types=[(ENTRADA, StringTensorType([None, 1]))],
        target_opset=OPSET,
        options={id(classificador): {"zipmap": False}},
    )
    if vetorizador.sublinear_tf:
        _corrigir_sublinear_tf(modelo)
    onnx.checker.check_model(modelo)
    return modelo


def _com_calibracao_densa(pipeline: Pipeline) -> Pipeline:
    """Copia do pipeline com os calibradores isotonicos densificados (o original e intocado).

    O skl2onnx converte a calibracao isotonica como "valor do limiar mais proximo", enquanto o
    scikit-learn interpola linearmente entre os limiares. Com poucos limiares a diferenca e
    grande (visto: ate 0,11 num modelo pequeno); com os do modelo real e pequena (~0,014).
    Aqui cada calibrador ganha pontos intermediarios sobre a propria reta de interpolacao,
    de modo que o limiar mais proximo passe a dar, ate `PASSO_CALIBRACAO`, o mesmo valor.
    """
    copia = copy.deepcopy(pipeline)
    classificador = copia.steps[-1][1]
    if isinstance(classificador, CalibratedClassifierCV) and classificador.method == "isotonic":
        for calibrado in classificador.calibrated_classifiers_:
            for calibrador in calibrado.calibrators:
                _densificar_calibrador(calibrador)
    return copia


def _densificar_calibrador(calibrador: IsotonicRegression) -> None:
    """Insere pontos nos trechos inclinados, sem alterar a funcao (ver `_com_calibracao_densa`)."""
    x, y = calibrador.X_thresholds_, calibrador.y_thresholds_
    novos_x = [x[:-1]]
    for inicio, fim, y_inicio, y_fim in zip(x[:-1], x[1:], y[:-1], y[1:], strict=True):
        pedacos = max(1, int(np.ceil(abs(y_fim - y_inicio) / PASSO_CALIBRACAO)))
        novos_x.append(np.linspace(inicio, fim, pedacos + 1)[:-1] if pedacos > 1 else [])
    denso_x = np.unique(np.concatenate([*novos_x, x[-1:]]))
    denso_y = np.interp(denso_x, x, y)
    for nome_x, nome_y in (("X_thresholds_", "y_thresholds_"), ("_necessary_X_", "_necessary_y_")):
        setattr(calibrador, nome_x, denso_x)
        setattr(calibrador, nome_y, denso_y)


def _corrigir_sublinear_tf(modelo: onnx.ModelProto) -> None:
    """Troca `log(contagem + 1)` por `1 + log(contagem)` (e 0 onde a contagem e 0).

    O skl2onnx implementa o `sublinear_tf` do scikit-learn como `log(contagem + 1)`, formula
    diferente de `1 + log(contagem)`; sem a troca, os TF-IDF divergem em todos os documentos
    (medido: ate 0,046) e cerca de 0,3% das decisoes de urgencia mudam.

    Raises:
        ConversaoOnnxError: o grafo nao tem o trecho esperado (ex.: o skl2onnx foi corrigido
            ou mudou o formato); a conversao nao deve seguir sem revisao.
    """
    grafo = modelo.graph
    constantes = {i.name: numpy_helper.to_array(i) for i in grafo.initializer}
    produtor = {saida: no for no in grafo.node for saida in no.output}
    alvos = []
    for log in (no for no in grafo.node if no.op_type == "Log"):
        soma = produtor.get(log.input[0])
        if soma is None or soma.op_type != "Add":
            continue
        contagem = [e for e in soma.input if e not in constantes]
        uns = [e for e in soma.input if e in constantes and np.all(constantes[e] == 1)]
        if len(contagem) == 1 and len(uns) == 1:
            alvos.append((soma, log, contagem[0]))
    if len(alvos) != 1:
        raise ConversaoOnnxError(
            f"Esperado exatamente um trecho log(contagem + 1) no grafo, achei {len(alvos)}."
        )
    soma, log, contagem = alvos[0]
    grafo.initializer.extend(
        [
            numpy_helper.from_array(np.array(0, np.float32), "sublinear_zero"),
            numpy_helper.from_array(np.array(1, np.float32), "sublinear_um"),
        ]
    )
    novos = [
        helper.make_node("Greater", [contagem, "sublinear_zero"], ["sublinear_positivo"]),
        helper.make_node(
            "Cast", ["sublinear_positivo"], ["sublinear_mascara"], to=TensorProto.FLOAT
        ),
        helper.make_node("Max", [contagem, "sublinear_um"], ["sublinear_limitado"]),
        helper.make_node("Log", ["sublinear_limitado"], ["sublinear_log"]),
        helper.make_node("Add", ["sublinear_log", "sublinear_um"], ["sublinear_um_mais_log"]),
        helper.make_node("Mul", ["sublinear_um_mais_log", "sublinear_mascara"], [log.output[0]]),
    ]
    nos = list(grafo.node)
    posicao = nos.index(soma)
    restante = [no for no in nos if no is not soma and no is not log]
    restante[posicao:posicao] = novos
    del grafo.node[:]
    grafo.node.extend(restante)


def verificar_paridade(
    pipeline: Pipeline,
    caminho_onnx: Path,
    textos: Sequence[str],
    limiar_urgente: float,
    diferenca_maxima: float = DIFERENCA_MAXIMA_PROBABILIDADE,
    divergencia_maxima: float = DIVERGENCIA_MAXIMA_URGENCIA,
) -> dict[str, Any]:
    """Compara o ONNX com o pipeline original em `textos` e aplica as tolerancias.

    O ONNX e chamado laudo a laudo, pelo mesmo caminho que a API usa em producao. A conta de
    urgencia e a mesma do `Classificador` (agregacao por nivel e limiar).

    Args:
        textos: laudos brutos; sao normalizados aqui. Textos sem termos uteis sao ignorados.

    Raises:
        ParidadeOnnxError: probabilidade ou decisao de urgencia alem da tolerancia.
    """
    normalizados = [t for t in (normalizar(texto) for texto in textos) if t]
    if not normalizados:
        raise ValueError("Nenhum texto utilizavel para verificar a paridade.")
    classes = [int(c) for c in pipeline.classes_]
    modelo = ModeloOnnx.abrir(caminho_onnx, classes)
    esperado = pipeline.predict_proba(normalizados)
    obtido = np.vstack([modelo.prever(texto) for texto in normalizados])

    def urgencias(probabilidades: np.ndarray) -> list[Any]:
        return [
            decidir_urgencia(
                agregar_probabilidades({c: float(p) for c, p in zip(classes, linha, strict=True)}),
                limiar_urgente,
            )
            for linha in probabilidades
        ]

    diferencas = np.abs(obtido - esperado)
    divergentes = sum(a != b for a, b in zip(urgencias(esperado), urgencias(obtido), strict=True))
    relatorio = {
        "amostras": len(normalizados),
        "diferenca_maxima": round(float(diferencas.max()), 6),
        "diferenca_media": round(float(diferencas.mean()), 8),
        "urgencias_divergentes": int(divergentes),
        "fracao_divergente": round(divergentes / len(normalizados), 6),
        "tolerancia_diferenca": diferenca_maxima,
        "tolerancia_divergencia": divergencia_maxima,
    }
    if relatorio["diferenca_maxima"] > diferenca_maxima or (
        relatorio["fracao_divergente"] > divergencia_maxima
    ):
        raise ParidadeOnnxError(f"O ONNX nao reproduz o modelo original: {relatorio}")
    return relatorio


def exportar_onnx(settings: Settings, textos_teste: Sequence[str]) -> dict[str, Any]:
    """Converte `modelo.joblib` em `modelo.onnx` e registra o resultado nos metadados.

    O ONNX e gravado em arquivo temporario, verificado e so entao renomeado; os metadados
    (que referenciam o SHA-256 e a data do treino) sao trocados por ultimo. Assim um erro no
    meio nunca deixa um ONNX nao verificado no lugar do anterior.

    Args:
        textos_teste: laudos brutos do conjunto de teste, usados na verificacao de paridade.

    Returns:
        O bloco `onnx` gravado em `metadata.json`.

    Raises:
        ConversaoOnnxError, ParidadeOnnxError: conversao ou paridade reprovada (nada e gravado).
        FileNotFoundError: sem `modelo.joblib` e `metadata.json` em `artifacts_dir`.
    """
    metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    if metadata["versoes"]["scikit_learn"] != sklearn.__version__:
        raise ConversaoOnnxError(
            f"Modelo treinado com scikit-learn {metadata['versoes']['scikit_learn']}, "
            f"ambiente com {sklearn.__version__}."
        )
    pipeline = joblib.load(settings.model_path)
    modelo = converter(pipeline)

    temporario = settings.onnx_path.with_suffix(".onnx.novo")
    temporario.write_bytes(modelo.SerializeToString())
    try:
        # O texto com acentos garante que o caminho dos tokens nao-ASCII tambem e exercitado.
        textos = [*textos_teste, *(t + SUFIXO_NAO_ASCII for t in textos_teste[:200])]
        paridade = verificar_paridade(
            pipeline, temporario, textos, float(metadata["limiar_urgente"])
        )
        sha256 = hashlib.sha256(temporario.read_bytes()).hexdigest()
        os.replace(temporario, settings.onnx_path)
    finally:
        temporario.unlink(missing_ok=True)

    registro = {
        "arquivo": settings.onnx_path.name,
        "sha256": sha256,
        "treinado_em": metadata["treinado_em"],
        "classes": [int(c) for c in pipeline.classes_],
        "gerado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "versoes": {
            "skl2onnx": skl2onnx.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "opset": OPSET[""],
        },
        "paridade": paridade,
    }
    metadata["onnx"] = registro
    novo_metadata = settings.metadata_path.with_suffix(".json.novo")
    novo_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    os.replace(novo_metadata, settings.metadata_path)
    return registro


def main(argv: Sequence[str] | None = None) -> None:
    """Ponto de entrada: python -m triagem.onnx_export."""
    argparse.ArgumentParser(description="Exporta o modelo treinado para ONNX.").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings.from_env()
    try:
        teste = carregar_corpus(baixar_dataset(settings)["medical_tc_test.csv"])
        registro = exportar_onnx(settings, teste.textos)
    except (OSError, ValueError, RuntimeError) as erro:
        logger.error("Exportacao falhou: %s", erro)
        sys.exit(1)
    paridade = registro["paridade"]
    logger.info(
        "ONNX gravado em %s (%.1f MB). Paridade em %d laudos: diferenca maxima %.4f, "
        "urgencias divergentes %d.",
        settings.onnx_path,
        settings.onnx_path.stat().st_size / 1e6,
        paridade["amostras"],
        paridade["diferenca_maxima"],
        paridade["urgencias_divergentes"],
    )


if __name__ == "__main__":
    main()
