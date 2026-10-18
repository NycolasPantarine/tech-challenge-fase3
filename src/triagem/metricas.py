"""Metricas Prometheus da API.

Privacidade e cardinalidade: nenhuma metrica recebe o texto do laudo, e todos os rotulos tem
poucos valores possiveis (rota pelo TEMPLATE da rota, metodo, status, nivel de urgencia). Um
rotulo com valor livre (caminho cru, texto) criaria uma serie nova por requisicao e
estouraria a memoria do Prometheus.

Cada aplicacao tem o proprio registro: evita erro de metrica duplicada quando varias
aplicacoes sao criadas no mesmo processo (testes) e nao depende do registro global.
Pressupoe uma unica worker do uvicorn; com varias, seria preciso o modo multiprocesso do
`prometheus_client`.
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    disable_created_metrics,
)
from prometheus_client import generate_latest as _gerar_texto

# Remove as series auxiliares `*_created` (instante de criacao de cada serie): o dashboard
# nao as usa e elas dobrariam o numero de series de cada contador e histograma.
disable_created_metrics()

# Faixa em que a API opera (a inferencia fica perto de 10 ms): de 5 ms a 2,5 s.
BUCKETS_HTTP = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
BUCKETS_INFERENCIA = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
BUCKETS_PROBABILIDADE = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

ROTULO_ROTA_DESCONHECIDA = "desconhecida"


class Metricas:
    """Conjunto de metricas de uma aplicacao e o registro que as guarda."""

    def __init__(self) -> None:
        self.registro = CollectorRegistry()
        self._requisicoes = Counter(
            "http_requests_total",
            "Requisicoes HTTP atendidas.",
            ["method", "route", "status"],
            registry=self.registro,
        )
        self._duracao_http = Histogram(
            "http_request_duration_seconds",
            "Duracao das requisicoes HTTP, vista pelo servidor.",
            ["route"],
            buckets=BUCKETS_HTTP,
            registry=self.registro,
        )
        self._duracao_inferencia = Histogram(
            "triagem_inferencia_duration_seconds",
            "Duracao so da classificacao (normalizacao do texto e modelo).",
            buckets=BUCKETS_INFERENCIA,
            registry=self.registro,
        )
        self._predicoes = Counter(
            "triagem_predicoes_total",
            "Laudos classificados, por nivel de urgencia.",
            ["urgencia"],
            registry=self.registro,
        )
        self._probabilidade_urgente = Histogram(
            "triagem_probabilidade_urgente",
            "Distribuicao da probabilidade de urgencia 'urgente' nas predicoes.",
            buckets=BUCKETS_PROBABILIDADE,
            registry=self.registro,
        )
        self._modelo = Gauge(
            "triagem_modelo_info",
            "Versao do modelo em uso (valor sempre 1).",
            ["versao"],
            registry=self.registro,
        )

    def registrar_http(self, metodo: str, rota: str, status: int, segundos: float) -> None:
        """Conta uma requisicao HTTP e registra sua duracao."""
        self._requisicoes.labels(method=metodo, route=rota, status=str(status)).inc()
        self._duracao_http.labels(route=rota).observe(segundos)

    def registrar_predicao(
        self, urgencia: str, probabilidade_urgente: float, segundos: float
    ) -> None:
        """Registra uma classificacao concluida."""
        self._predicoes.labels(urgencia=urgencia).inc()
        self._probabilidade_urgente.observe(probabilidade_urgente)
        self._duracao_inferencia.observe(segundos)

    def definir_modelo(self, versao: str) -> None:
        """Informa a versao do modelo carregado."""
        self._modelo.labels(versao=versao).set(1)

    def exportar(self) -> tuple[bytes, str]:
        """Devolve o texto no formato do Prometheus e o seu Content-Type."""
        return _gerar_texto(self.registro), CONTENT_TYPE_LATEST
