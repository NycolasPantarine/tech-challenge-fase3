"""Metricas de avaliacao em dois niveis: categoria clinica (5) e urgencia (3)."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from triagem.urgencia import CATEGORIA_PARA_URGENCIA, Urgencia, decidir_urgencia

NIVEIS: tuple[Urgencia, ...] = (Urgencia.NORMAL, Urgencia.ATENCAO, Urgencia.URGENTE)


def agregar_por_urgencia(probabilidades: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    """Soma as probabilidades das categorias de cada nivel.

    Args:
        probabilidades: matriz (n_amostras, n_categorias), colunas na ordem de `classes`.
        classes: categoria correspondente a cada coluna.

    Returns:
        Matriz (n_amostras, 3) com colunas na ordem de NIVEIS.
    """
    agregado = np.zeros((probabilidades.shape[0], len(NIVEIS)))
    for coluna, categoria in enumerate(classes):
        agregado[:, NIVEIS.index(CATEGORIA_PARA_URGENCIA[categoria])] += probabilidades[:, coluna]
    return agregado


def prever_urgencias(prob_urgencia: np.ndarray, limiar_urgente: float | None) -> list[Urgencia]:
    """Aplica a regra de decisao (ver `decidir_urgencia`) a cada linha."""
    return [
        decidir_urgencia(dict(zip(NIVEIS, linha, strict=True)), limiar_urgente)
        for linha in prob_urgencia
    ]


def escolher_limiar_urgente(
    urgente_real: np.ndarray, prob_urgente: np.ndarray, recall_alvo: float
) -> float:
    """Maior limiar cujo recall de `urgente` ainda atinge o alvo.

    Quanto maior o limiar, maior a precisao; por isso escolhe-se o maior que cumpre o recall.
    Se nenhum limiar atinge o alvo, retorna 0.0 (todo caso vira urgente).

    Raises:
        ValueError: se o alvo estiver fora de (0, 1].
    """
    if not 0.0 < recall_alvo <= 1.0:
        raise ValueError(f"Recall alvo deve estar em (0, 1]: {recall_alvo!r}")
    _, recall, limiares = precision_recall_curve(urgente_real, prob_urgente)
    atingem = np.flatnonzero(recall[:-1] >= recall_alvo)
    return float(limiares[atingem[-1]]) if atingem.size else 0.0


def avaliar(
    categorias_reais: Sequence[int],
    probabilidades: np.ndarray,
    classes: Sequence[int],
    limiar_urgente: float | None,
) -> dict[str, Any]:
    """Calcula metricas por categoria (argmax) e por urgencia (com ou sem limiar)."""
    previstas_cat = [classes[i] for i in probabilidades.argmax(axis=1)]
    reais_urg = [CATEGORIA_PARA_URGENCIA[c].value for c in categorias_reais]
    prob_urg = agregar_por_urgencia(probabilidades, classes)
    previstas_urg = [u.value for u in prever_urgencias(prob_urg, limiar_urgente)]
    alvo = Urgencia.URGENTE.value

    return {
        "acuracia_categoria": accuracy_score(categorias_reais, previstas_cat),
        "f1_macro_categoria": f1_score(categorias_reais, previstas_cat, average="macro"),
        "acuracia_urgencia": accuracy_score(reais_urg, previstas_urg),
        "f1_macro_urgencia": f1_score(reais_urg, previstas_urg, average="macro"),
        "recall_urgente": recall_score(
            reais_urg, previstas_urg, labels=[alvo], average=None, zero_division=0
        )[0],
        "precisao_urgente": precision_score(
            reais_urg, previstas_urg, labels=[alvo], average=None, zero_division=0
        )[0],
        "limiar_urgente": limiar_urgente,
        "niveis": [nivel.value for nivel in NIVEIS],
        "matriz_confusao_urgencia": confusion_matrix(
            reais_urg, previstas_urg, labels=[n.value for n in NIVEIS]
        ).tolist(),
    }
