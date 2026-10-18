from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from triagem.carga import (
    TEXTOS_INVALIDOS,
    ResumoCarga,
    escolher_texto,
    executar,
    formatar_resumo,
)

TEXTOS = ["cardiac aortic coronary", "tumor carcinoma neoplasm", "neuron cerebral seizure"]


class _RelogioFalso:
    """Relogio controlado: `dormir` avanca o tempo, entao o teste nao espera de verdade."""

    def __init__(self) -> None:
        self.agora = 0.0

    def __call__(self) -> float:
        return self.agora

    def dormir(self, segundos: float) -> None:
        self.agora += segundos


def _executar(
    enviar: Callable[[str], int], relogio: _RelogioFalso, **parametros: Any
) -> ResumoCarga:
    padrao: dict[str, Any] = {
        "textos": TEXTOS,
        "duracao": 10.0,
        "taxa": 2.0,
        "proporcao_invalidas": 0.0,
        "semente": 1,
    }
    return executar(enviar, **{**padrao, **parametros}, relogio=relogio, dormir=relogio.dormir)


def test_sem_invalidas_so_sorteia_laudos_reais() -> None:
    sorteio = np.random.default_rng(1)

    escolhidos = {escolher_texto(sorteio, TEXTOS, 0.0) for _ in range(200)}

    assert escolhidos <= set(TEXTOS)


def test_so_invalidas_nunca_sorteia_laudo_real() -> None:
    sorteio = np.random.default_rng(1)

    escolhidos = {escolher_texto(sorteio, TEXTOS, 1.0) for _ in range(200)}

    assert escolhidos == set(TEXTOS_INVALIDOS)


def test_sorteio_e_repetivel_para_a_mesma_semente() -> None:
    primeiro = [escolher_texto(np.random.default_rng(7), TEXTOS, 0.5) for _ in range(5)]
    segundo = [escolher_texto(np.random.default_rng(7), TEXTOS, 0.5) for _ in range(5)]

    assert primeiro == segundo


def test_envia_na_taxa_pedida_durante_a_duracao() -> None:
    relogio = _RelogioFalso()
    recebidos: list[str] = []

    resumo = _executar(lambda texto: recebidos.append(texto) or 200, relogio)

    assert resumo.enviadas == 20  # 2 req/s x 10 s
    assert resumo.por_status == {200: 20}
    assert relogio.agora == pytest.approx(10.0)


def test_conta_cada_status_recebido() -> None:
    relogio = _RelogioFalso()

    resumo = _executar(
        lambda texto: 422 if texto in TEXTOS_INVALIDOS else 200,
        relogio,
        proporcao_invalidas=0.5,
        duracao=50.0,
    )

    assert set(resumo.por_status) == {200, 422}
    assert resumo.enviadas == 100


def test_falha_de_conexao_e_contada_e_a_carga_continua() -> None:
    relogio = _RelogioFalso()

    def enviar(texto: str) -> int:
        raise ConnectionRefusedError("api fora do ar")

    resumo = _executar(enviar, relogio)

    assert resumo.falhas_conexao == 20
    assert resumo.enviadas == 0


def test_ctrl_c_encerra_e_devolve_o_que_foi_contado() -> None:
    relogio = _RelogioFalso()
    chamadas = 0

    def enviar(texto: str) -> int:
        nonlocal chamadas
        chamadas += 1
        if chamadas == 4:
            raise KeyboardInterrupt
        return 200

    resumo = _executar(enviar, relogio)

    assert resumo.interrompida
    assert resumo.enviadas == 3


@pytest.mark.parametrize(
    "parametros",
    [
        {"textos": []},
        {"duracao": 0},
        {"taxa": -1},
        {"proporcao_invalidas": 1.5},
        {"proporcao_invalidas": -0.1},
    ],
)
def test_rejeita_parametros_invalidos(parametros: dict[str, Any]) -> None:
    relogio = _RelogioFalso()

    with pytest.raises(ValueError):
        _executar(lambda texto: 200, relogio, **parametros)


def test_resumo_formatado_lista_status_em_ordem() -> None:
    relogio = _RelogioFalso()
    resumo = _executar(lambda texto: 422 if len(texto) % 2 else 200, relogio)

    texto = formatar_resumo(resumo)

    assert texto.startswith("enviadas=20 [")
    assert "falhas_de_conexao=0" in texto
