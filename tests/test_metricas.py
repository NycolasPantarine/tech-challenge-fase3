from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from triagem.api import criar_app
from triagem.classificador import Classificador
from triagem.config import Settings
from triagem.metricas import Metricas

TEXTO_CARDIACO = "Cardiac coronary infarction with aortic involvement"


@pytest.fixture(scope="module")
def classificador(settings_com_modelo: Settings) -> Classificador:
    return Classificador.carregar(settings_com_modelo)


@pytest.fixture
def cliente(classificador: Classificador) -> Iterator[TestClient]:
    """Aplicacao nova por teste: cada uma tem o proprio registro, entao as contagens isolam."""
    with TestClient(criar_app(classificador)) as cliente:
        yield cliente


def _valor(cliente: TestClient, nome: str, rotulos: dict[str, str] | None = None) -> float:
    return cliente.app.state.metricas.registro.get_sample_value(nome, rotulos or {}) or 0.0


def test_metrics_expoe_o_formato_do_prometheus(cliente: TestClient) -> None:
    resposta = cliente.get("/metrics")

    assert resposta.status_code == 200
    assert resposta.headers["content-type"].startswith("text/plain")
    assert "# TYPE http_requests_total counter" in resposta.text
    assert "# TYPE triagem_inferencia_duration_seconds histogram" in resposta.text


def test_metrics_nao_expoe_series_created(cliente: TestClient) -> None:
    cliente.post("/predict", json={"texto": TEXTO_CARDIACO})

    assert "_created" not in cliente.get("/metrics").text


def test_metrics_nao_aparece_na_documentacao_da_api(cliente: TestClient) -> None:
    assert "/metrics" not in cliente.get("/openapi.json").json()["paths"]


def test_predict_conta_requisicao_predicao_e_inferencia(cliente: TestClient) -> None:
    cliente.post("/predict", json={"texto": TEXTO_CARDIACO})

    requisicao = {"method": "POST", "route": "/predict", "status": "200"}
    assert _valor(cliente, "http_requests_total", requisicao) == 1
    assert _valor(cliente, "http_request_duration_seconds_count", {"route": "/predict"}) == 1
    assert _valor(cliente, "triagem_predicoes_total", {"urgencia": "urgente"}) == 1
    assert _valor(cliente, "triagem_inferencia_duration_seconds_count") == 1
    assert _valor(cliente, "triagem_probabilidade_urgente_count") == 1


def test_modelo_em_uso_aparece_como_gauge(
    cliente: TestClient, classificador: Classificador
) -> None:
    assert _valor(cliente, "triagem_modelo_info", {"versao": classificador.versao}) == 1


def test_entrada_invalida_conta_422_e_nao_conta_predicao(cliente: TestClient) -> None:
    cliente.post("/predict", json={"texto": ""})
    cliente.post("/predict", json={"texto": "!!! the of and ..."})

    requisicao = {"method": "POST", "route": "/predict", "status": "422"}
    assert _valor(cliente, "http_requests_total", requisicao) == 2
    assert _valor(cliente, "triagem_inferencia_duration_seconds_count") == 0
    for urgencia in ("normal", "atencao", "urgente"):
        assert _valor(cliente, "triagem_predicoes_total", {"urgencia": urgencia}) == 0


def test_rota_usa_o_template_e_nunca_o_caminho_cru(cliente: TestClient) -> None:
    cliente.get("/caminho-aleatorio-123")
    cliente.get("/outro-caminho-456")

    texto = cliente.get("/metrics").text
    assert 'route="desconhecida"' in texto
    assert "caminho-aleatorio" not in texto
    assert "outro-caminho" not in texto


def test_a_leitura_de_metrics_nao_infla_a_vazao(cliente: TestClient) -> None:
    for _ in range(3):
        cliente.get("/metrics")

    assert 'route="/metrics"' not in cliente.get("/metrics").text


def test_erro_interno_e_contado_como_500(classificador: Classificador) -> None:
    class Quebrado:
        versao = "quebrado@teste"
        limiar_urgente = 0.5

        def classificar(self, texto: str) -> object:
            raise RuntimeError("falha simulada")

    with TestClient(criar_app(Quebrado()), raise_server_exceptions=False) as cliente:
        resposta = cliente.post("/predict", json={"texto": TEXTO_CARDIACO})

        assert resposta.status_code == 500
        requisicao = {"method": "POST", "route": "/predict", "status": "500"}
        assert _valor(cliente, "http_requests_total", requisicao) == 1


def test_conteudo_do_laudo_nunca_aparece_nas_metricas(cliente: TestClient) -> None:
    cliente.post("/predict", json={"texto": "paciente identificavel com infarction sigiloso"})

    texto = cliente.get("/metrics").text
    assert "sigiloso" not in texto
    assert "paciente" not in texto


def test_cada_aplicacao_tem_o_proprio_registro(classificador: Classificador) -> None:
    primeira, segunda = Metricas(), Metricas()

    primeira.registrar_http("GET", "/health", 200, 0.01)

    assert (
        primeira.registro.get_sample_value(
            "http_requests_total", {"method": "GET", "route": "/health", "status": "200"}
        )
        == 1
    )
    assert (
        segunda.registro.get_sample_value(
            "http_requests_total", {"method": "GET", "route": "/health", "status": "200"}
        )
        is None
    )
