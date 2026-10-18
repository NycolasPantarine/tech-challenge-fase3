"""Garante que o dashboard do Grafana e a configuracao da stack nao saiam de sincronia com a API.

O Grafana nao roda nos testes; o que se verifica aqui e que cada metrica citada numa consulta
do dashboard existe de fato no `/metrics` da API (renomear uma metrica sem atualizar o JSON
deixaria o painel vazio sem nenhum erro).
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from triagem.api import criar_app
from triagem.classificador import Classificador
from triagem.config import Settings

RAIZ = Path(__file__).resolve().parent.parent
DASHBOARD = RAIZ / "monitoring" / "grafana" / "dashboards" / "triagem.json"
DATASOURCE = RAIZ / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
PROMETHEUS = RAIZ / "monitoring" / "prometheus.yml"
COMPOSE = RAIZ / "docker-compose.yml"

MINIMO_DE_PAINEIS = 3  # exigencia do desafio
PADRAO_METRICA = re.compile(r"\b((?:http|triagem)_[a-z_]+)")
SUFIXOS_DE_HISTOGRAMA = ("_bucket", "_sum", "_count")


@pytest.fixture(scope="module")
def dashboard() -> dict[str, Any]:
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metricas_expostas(settings_com_modelo: Settings) -> set[str]:
    with TestClient(criar_app(Classificador.carregar(settings_com_modelo))) as cliente:
        cliente.post("/predict", json={"texto": "Cardiac coronary infarction"})
        texto = cliente.get("/metrics").text
    return set(re.findall(r"^# TYPE (\S+) ", texto, flags=re.MULTILINE))


def _alvos(dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    return [alvo for painel in dashboard["panels"] for alvo in painel["targets"]]


def _metricas_citadas(expressao: str) -> set[str]:
    nomes = set()
    for nome in PADRAO_METRICA.findall(expressao):
        for sufixo in SUFIXOS_DE_HISTOGRAMA:
            nome = nome.removesuffix(sufixo)
        nomes.add(nome)
    return nomes


def test_dashboard_tem_ao_menos_tres_paineis(dashboard: dict[str, Any]) -> None:
    assert len(dashboard["panels"]) >= MINIMO_DE_PAINEIS


def test_paineis_tem_ids_e_titulos_unicos(dashboard: dict[str, Any]) -> None:
    ids = [painel["id"] for painel in dashboard["panels"]]
    titulos = [painel["title"] for painel in dashboard["panels"]]

    assert len(set(ids)) == len(ids)
    assert len(set(titulos)) == len(titulos)


def test_todo_painel_tem_ao_menos_uma_consulta(dashboard: dict[str, Any]) -> None:
    assert all(painel["targets"] for painel in dashboard["panels"])


def test_toda_metrica_do_dashboard_existe_na_api(
    dashboard: dict[str, Any], metricas_expostas: set[str]
) -> None:
    citadas = set().union(*(_metricas_citadas(alvo["expr"]) for alvo in _alvos(dashboard)))

    assert citadas, "nenhuma metrica encontrada nas consultas"
    assert citadas <= metricas_expostas, f"metricas inexistentes: {citadas - metricas_expostas}"


def test_consultas_usam_o_datasource_provisionado(dashboard: dict[str, Any]) -> None:
    uid = re.search(r"^\s*uid:\s*(\S+)", DATASOURCE.read_text(encoding="utf-8"), re.MULTILINE)

    assert uid is not None
    for painel in dashboard["panels"]:
        assert painel["datasource"]["uid"] == uid.group(1)
        assert all(alvo["datasource"]["uid"] == uid.group(1) for alvo in painel["targets"])


def test_prometheus_coleta_a_api_pelo_nome_e_porta_do_compose() -> None:
    assert "api:8000" in PROMETHEUS.read_text(encoding="utf-8")
    compose = COMPOSE.read_text(encoding="utf-8")
    assert re.search(r"^  api:", compose, re.MULTILINE)
    assert "8000" in compose


def test_compose_exige_senha_do_grafana_e_nao_publica_portas_na_rede() -> None:
    compose = COMPOSE.read_text(encoding="utf-8")

    assert "${GRAFANA_ADMIN_PASSWORD:?" in compose
    portas = re.findall(r'^\s+- "([^"]+:\d+)"', compose, re.MULTILINE)
    assert portas
    assert all(porta.startswith("127.0.0.1:") for porta in portas)
