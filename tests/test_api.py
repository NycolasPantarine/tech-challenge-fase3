import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from triagem.api import TAMANHO_MAXIMO_TEXTO, criar_app
from triagem.classificador import Classificador, ModeloIndisponivelError
from triagem.config import Settings

TEXTO_CARDIACO = "Cardiac coronary infarction with aortic involvement"


@pytest.fixture(scope="module")
def cliente(settings_com_modelo: Settings) -> Iterator[TestClient]:
    classificador = Classificador.carregar(settings_com_modelo)
    with TestClient(criar_app(classificador)) as cliente:
        yield cliente


def test_health_informa_versao_do_modelo(cliente: TestClient) -> None:
    resposta = cliente.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {
        "status": "ok",
        "versao_modelo": "naive_bayes_calibrado@2026-01-01T00:00:00+00:00",
    }


def test_predict_retorna_urgencia_categoria_e_probabilidades(cliente: TestClient) -> None:
    resposta = cliente.post("/predict", json={"texto": TEXTO_CARDIACO})

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["urgencia"] == "urgente"
    assert corpo["categoria"]["id"] == 4
    assert corpo["categoria"]["nome"] == "cardiovascular diseases"
    assert sum(corpo["probabilidades_urgencia"].values()) == pytest.approx(1.0)
    assert set(corpo["probabilidades_urgencia"]) == {"normal", "atencao", "urgente"}
    assert corpo["limiar_urgente"] == 0.5
    assert corpo["versao_modelo"].startswith("naive_bayes_calibrado@")


def test_resposta_inclui_id_da_requisicao(cliente: TestClient) -> None:
    resposta = cliente.get("/health")

    assert len(resposta.headers["X-Request-ID"]) == 12


@pytest.mark.parametrize(
    "corpo",
    [
        {},
        {"texto": ""},
        {"texto": "     "},
        {"texto": "x" * (TAMANHO_MAXIMO_TEXTO + 1)},
        {"texto": 123},
    ],
)
def test_predict_rejeita_entrada_invalida(cliente: TestClient, corpo: dict[str, object]) -> None:
    resposta = cliente.post("/predict", json=corpo)

    assert resposta.status_code == 422


def test_predict_rejeita_texto_sem_termos_uteis(cliente: TestClient) -> None:
    resposta = cliente.post("/predict", json={"texto": "!!! the of and ..."})

    assert resposta.status_code == 422
    assert "termos uteis" in resposta.json()["detail"]


def test_conteudo_do_laudo_nunca_aparece_no_log(
    cliente: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    texto_sigiloso = "paciente identificavel com infarction coronary sigiloso"

    with caplog.at_level(logging.DEBUG):
        cliente.post("/predict", json={"texto": texto_sigiloso})

    assert "sigiloso" not in caplog.text
    assert "paciente" not in caplog.text


def test_api_nao_sobe_sem_artefato_do_modelo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIAGEM_ARTIFACTS_DIR", str(tmp_path / "inexistente"))

    with pytest.raises(ModeloIndisponivelError), TestClient(criar_app()):
        pass
