"""Comparacao de modelos, treino final e geracao do artefato.

Fluxo: valida o dataset, compara os candidatos por validacao cruzada no TREINO, escolhe o
melhor por F1 macro de urgencia, calibra o limiar de `urgente` para um recall alvo, treina o
modelo final em todo o treino e avalia UMA vez no teste. O teste nunca participa da escolha.
"""

import argparse
import hashlib
import json
import logging
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from triagem.avaliacao import agregar_por_urgencia, avaliar, escolher_limiar_urgente
from triagem.config import Settings
from triagem.corpus import Corpus, carregar_corpus
from triagem.dados import baixar_dataset
from triagem.modelo import RANDOM_STATE, TipoModelo, construir_pipeline, preparar_para_inferencia
from triagem.texto import normalizar
from triagem.urgencia import CATEGORIA_PARA_URGENCIA, CATEGORIAS, Urgencia

logger = logging.getLogger(__name__)

RECALL_ALVO_PADRAO = 0.80
FOLDS_PADRAO = 3


@dataclass(frozen=True)
class ResultadoCandidato:
    """Resultado da validacao cruzada de um candidato (probabilidades fora da dobra)."""

    tipo: TipoModelo
    metricas: dict[str, Any]
    probabilidades: np.ndarray
    segundos: float


def normalizar_corpus(corpus: Corpus) -> list[str]:
    """Aplica a normalizacao de texto a todos os documentos."""
    return [normalizar(texto) for texto in corpus.textos]


def comparar_candidatos(
    textos: Sequence[str],
    categorias: Sequence[int],
    tipos: Sequence[TipoModelo],
    folds: int,
) -> list[ResultadoCandidato]:
    """Avalia cada candidato por validacao cruzada estratificada, sem usar o teste."""
    validacao = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    classes = sorted(CATEGORIAS)
    resultados = []
    for tipo in tipos:
        inicio = time.perf_counter()
        probabilidades = cross_val_predict(
            construir_pipeline(tipo, n_jobs=-1),
            list(textos),
            list(categorias),
            cv=validacao,
            method="predict_proba",
        )
        segundos = time.perf_counter() - inicio
        metricas = avaliar(categorias, probabilidades, classes, limiar_urgente=None)
        logger.info(
            "%-12s F1 urgencia=%.3f  recall urgente=%.3f  acuracia categoria=%.3f  (%.0fs)",
            tipo.value,
            metricas["f1_macro_urgencia"],
            metricas["recall_urgente"],
            metricas["acuracia_categoria"],
            segundos,
        )
        resultados.append(ResultadoCandidato(tipo, metricas, probabilidades, segundos))
    return resultados


def selecionar_melhor(resultados: Sequence[ResultadoCandidato]) -> ResultadoCandidato:
    """Escolhe o candidato de maior F1 macro de urgencia (empate: o primeiro da lista)."""
    return max(resultados, key=lambda r: r.metricas["f1_macro_urgencia"])


def _sha256_arquivo(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def executar_treino(
    settings: Settings,
    treino: Corpus,
    teste: Corpus,
    tipos: Sequence[TipoModelo] = tuple(TipoModelo),
    folds: int = FOLDS_PADRAO,
    recall_alvo: float = RECALL_ALVO_PADRAO,
    hash_treino: str = "",
) -> dict[str, Any]:
    """Executa o fluxo completo e grava modelo e metadados em settings.artifacts_dir."""
    textos_treino = normalizar_corpus(treino)
    textos_teste = normalizar_corpus(teste)

    resultados = comparar_candidatos(textos_treino, treino.categorias, tipos, folds)
    melhor = selecionar_melhor(resultados)
    logger.info("Escolhido: %s", melhor.tipo.value)

    classes = sorted(CATEGORIAS)
    prob_urgente_cv = agregar_por_urgencia(melhor.probabilidades, classes)[:, -1]
    urgente_real = np.array(
        [CATEGORIA_PARA_URGENCIA[c] is Urgencia.URGENTE for c in treino.categorias]
    )
    limiar = escolher_limiar_urgente(urgente_real, prob_urgente_cv, recall_alvo)
    metricas_cv_com_limiar = avaliar(treino.categorias, melhor.probabilidades, classes, limiar)

    pipeline = construir_pipeline(melhor.tipo, n_jobs=-1)
    pipeline.fit(textos_treino, treino.categorias)
    preparar_para_inferencia(pipeline)
    metricas_teste = avaliar(
        teste.categorias, pipeline.predict_proba(textos_teste), list(pipeline.classes_), limiar
    )
    metricas_teste_argmax = avaliar(
        teste.categorias, pipeline.predict_proba(textos_teste), list(pipeline.classes_), None
    )

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, settings.model_path)
    metadata = {
        "tipo_modelo": melhor.tipo.value,
        "treinado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "versoes": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "sha256_dados_treino": hash_treino,
        "n_treino": len(treino),
        "n_teste": len(teste),
        "recall_alvo_urgente": recall_alvo,
        "limiar_urgente": limiar,
        "classes": [int(c) for c in pipeline.classes_],
        "candidatos_cv": {
            r.tipo.value: {**r.metricas, "segundos": round(r.segundos, 1)} for r in resultados
        },
        "melhor_cv_com_limiar": metricas_cv_com_limiar,
        "teste_com_limiar": metricas_teste,
        "teste_argmax": metricas_teste_argmax,
        "mapeamento_urgencia": {str(c): u.value for c, u in CATEGORIA_PARA_URGENCIA.items()},
    }
    settings.metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    """Ponto de entrada: python -m triagem.treino."""
    parser = argparse.ArgumentParser(description="Treina e avalia o classificador de urgencia.")
    parser.add_argument("--folds", type=int, default=FOLDS_PADRAO)
    parser.add_argument("--recall-alvo", type=float, default=RECALL_ALVO_PADRAO)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings.from_env()
    caminhos = baixar_dataset(settings)
    treino_csv = caminhos["medical_tc_train.csv"]
    metadata = executar_treino(
        settings,
        carregar_corpus(treino_csv),
        carregar_corpus(caminhos["medical_tc_test.csv"]),
        folds=args.folds,
        recall_alvo=args.recall_alvo,
        hash_treino=_sha256_arquivo(treino_csv),
    )
    teste = metadata["teste_com_limiar"]
    logger.info(
        "TESTE (limiar=%.3f): F1 urgencia=%.3f  recall urgente=%.3f  precisao urgente=%.3f",
        metadata["limiar_urgente"],
        teste["f1_macro_urgencia"],
        teste["recall_urgente"],
        teste["precisao_urgente"],
    )
    logger.info("Modelo: %s | Metadados: %s", settings.model_path, settings.metadata_path)


if __name__ == "__main__":
    main()
