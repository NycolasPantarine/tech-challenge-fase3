import numpy as np
import pytest

from triagem.avaliacao import (
    NIVEIS,
    agregar_por_urgencia,
    avaliar,
    escolher_limiar_urgente,
    prever_urgencias,
)
from triagem.urgencia import Urgencia, agregar_probabilidades

CLASSES = [1, 2, 3, 4, 5]


def test_agregar_por_urgencia_equivale_a_versao_escalar() -> None:
    linha = {1: 0.10, 2: 0.10, 3: 0.20, 4: 0.30, 5: 0.30}
    matriz = np.array([[linha[c] for c in CLASSES]])

    vetorizado = agregar_por_urgencia(matriz, CLASSES)[0]
    escalar = agregar_probabilidades(linha)

    for coluna, nivel in enumerate(NIVEIS):
        assert vetorizado[coluna] == pytest.approx(escalar[nivel])


def test_prever_urgencias_com_e_sem_limiar() -> None:
    prob = np.array([[0.2, 0.5, 0.3]])  # normal, atencao, urgente

    assert prever_urgencias(prob, None) == [Urgencia.ATENCAO]
    assert prever_urgencias(prob, 0.3) == [Urgencia.URGENTE]


def test_escolher_limiar_atinge_recall_alvo() -> None:
    real = np.array([True, True, True, True, False, False, False, False])
    prob = np.array([0.9, 0.8, 0.7, 0.4, 0.5, 0.3, 0.2, 0.1])

    limiar = escolher_limiar_urgente(real, prob, recall_alvo=0.75)

    previsto = prob >= limiar
    assert previsto[real].mean() >= 0.75
    assert limiar == pytest.approx(0.7)


def test_escolher_limiar_rejeita_alvo_invalido() -> None:
    with pytest.raises(ValueError, match="Recall alvo"):
        escolher_limiar_urgente(np.array([True, False]), np.array([0.9, 0.1]), recall_alvo=0.0)


def test_avaliar_modelo_perfeito() -> None:
    categorias = [1, 3, 5, 4]
    prob = np.eye(5)[[c - 1 for c in categorias]]

    metricas = avaliar(categorias, prob, CLASSES, limiar_urgente=None)

    assert metricas["acuracia_categoria"] == 1.0
    assert metricas["acuracia_urgencia"] == 1.0
    assert metricas["recall_urgente"] == 1.0
    assert metricas["precisao_urgente"] == 1.0
    assert metricas["matriz_confusao_urgencia"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
