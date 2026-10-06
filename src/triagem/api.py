"""API REST de triagem de urgencia em laudos medicos.

Execucao: uvicorn triagem.api:criar_app --factory

Privacidade: o texto do laudo e dado sensivel e NUNCA e registrado em log; so metodo, rota,
status, duracao e um identificador da requisicao.
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from triagem.classificador import Classificador, TextoSemConteudoError
from triagem.config import Settings

logger = logging.getLogger(__name__)

TAMANHO_MAXIMO_TEXTO = 10_000


class LaudoEntrada(BaseModel):
    """Laudo a ser classificado."""

    model_config = ConfigDict(str_strip_whitespace=True)

    texto: str = Field(min_length=1, max_length=TAMANHO_MAXIMO_TEXTO)


class CategoriaSaida(BaseModel):
    """Categoria clinica mais provavel."""

    id: int
    nome: str
    probabilidade: float


class PredicaoSaida(BaseModel):
    """Resultado da triagem."""

    urgencia: str
    probabilidades_urgencia: dict[str, float]
    categoria: CategoriaSaida
    limiar_urgente: float
    versao_modelo: str


class SaudeSaida(BaseModel):
    """Estado do servico."""

    status: str
    versao_modelo: str


def criar_app(classificador: Classificador | None = None) -> FastAPI:
    """Cria a aplicacao.

    Args:
        classificador: servico ja pronto (usado em testes). Se omitido, o modelo e carregado
            na inicializacao a partir de `Settings.from_env()`; sem artefato valido, a API
            nao sobe.
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
        app.state.classificador = classificador or Classificador.carregar(Settings.from_env())
        yield

    app = FastAPI(
        title="Triagem de urgencia em laudos medicos",
        description=(
            "Classifica laudos em `normal`, `atencao` ou `urgente`. "
            "Ferramenta de apoio a decisao: nao substitui avaliacao clinica."
        ),
        version="0.1.0",
        lifespan=ciclo_de_vida,
    )

    @app.middleware("http")
    async def registrar_requisicao(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        id_requisicao = uuid.uuid4().hex[:12]
        inicio = time.perf_counter()
        resposta = await call_next(request)
        resposta.headers["X-Request-ID"] = id_requisicao
        logger.info(
            "id=%s %s %s -> %s (%.1f ms)",
            id_requisicao,
            request.method,
            request.url.path,
            resposta.status_code,
            (time.perf_counter() - inicio) * 1000,
        )
        return resposta

    @app.get("/health", response_model=SaudeSaida)
    def health(request: Request) -> SaudeSaida:
        """Indica que o servico esta no ar com o modelo carregado."""
        return SaudeSaida(status="ok", versao_modelo=request.app.state.classificador.versao)

    @app.post("/predict", response_model=PredicaoSaida)
    def predict(entrada: LaudoEntrada, request: Request) -> PredicaoSaida:
        """Classifica a urgencia de um laudo."""
        servico: Classificador = request.app.state.classificador
        try:
            resultado = servico.classificar(entrada.texto)
        except TextoSemConteudoError as erro:
            raise HTTPException(status_code=422, detail=str(erro)) from erro
        return PredicaoSaida(
            urgencia=resultado.urgencia.value,
            probabilidades_urgencia={
                nivel.value: probabilidade
                for nivel, probabilidade in resultado.probabilidades_urgencia.items()
            },
            categoria=CategoriaSaida(
                id=resultado.categoria_id,
                nome=resultado.categoria_nome,
                probabilidade=resultado.categoria_probabilidade,
            ),
            limiar_urgente=servico.limiar_urgente,
            versao_modelo=servico.versao,
        )

    return app
