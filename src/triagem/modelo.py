"""Construcao dos pipelines candidatos (TF-IDF + classificador)."""

from enum import StrEnum

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


class TipoModelo(StrEnum):
    """Classificadores candidatos."""

    LOGISTICA = "logistica"
    NAIVE_BAYES_SIMPLES = "naive_bayes"
    NAIVE_BAYES = "naive_bayes_calibrado"
    FLORESTA = "floresta"


def _vetorizador() -> TfidfVectorizer:
    # lowercase=False e sem stop_words: a normalizacao e feita em triagem.texto.normalizar.
    return TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=3,
        max_features=30_000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def _classificador(tipo: TipoModelo, n_jobs: int) -> object:
    if tipo is TipoModelo.LOGISTICA:
        return LogisticRegression(C=5.0, max_iter=1000, class_weight="balanced")
    if tipo is TipoModelo.NAIVE_BAYES_SIMPLES:
        # Mantido na comparacao como referencia: mostra o efeito da calibracao (abaixo).
        return ComplementNB()
    if tipo is TipoModelo.NAIVE_BAYES:
        # O ComplementNB sozinho gera probabilidades pouco confiaveis (planas), o que distorce
        # a soma por nivel de urgencia e o limiar. A calibracao isotonica corrige isso.
        return CalibratedClassifierCV(ComplementNB(), method="isotonic", cv=3)
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=40,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
    )


def construir_pipeline(tipo: TipoModelo, n_jobs: int = 1) -> Pipeline:
    """Cria o pipeline nao treinado. A entrada deve ser texto ja normalizado."""
    return Pipeline([("tfidf", _vetorizador()), ("clf", _classificador(tipo, n_jobs))])


def preparar_para_inferencia(pipeline: Pipeline) -> Pipeline:
    """Forca inferencia em thread unica.

    Com n_jobs > 1 o Random Forest cria um pool de threads a cada chamada, o que penaliza
    a latencia de requisicoes individuais.
    """
    if "clf__n_jobs" in pipeline.get_params():
        pipeline.set_params(clf__n_jobs=1)
    return pipeline
