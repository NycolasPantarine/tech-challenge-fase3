"""Benchmark de latencia da classificacao, em dois niveis.

- **inferencia**: chama `Classificador.classificar` direto, no processo local. Mede so o modelo
  (normalizacao, TF-IDF, predicao e agregacao), sem FastAPI e sem rede.
- **http**: chama `POST /predict` de uma API em execucao (ex.: no container), com uma conexao
  persistente. Mede o que um cliente sente, incluindo FastAPI e a rede do Docker.

As requisicoes sao sequenciais: o objetivo e a latencia de uma requisicao isolada, nao o
throughput. As primeiras chamadas (aquecimento) sao descartadas porque a primeira costuma ser
muito mais lenta. Os textos sao laudos reais do conjunto de teste, amostrados com semente fixa.
O resultado depende da maquina; ele serve para comparar o modelo original com o otimizado
(Etapa 4) **na mesma maquina**, nao como valor absoluto de producao.

Uso: python -m triagem.benchmark [--url http://localhost:8010]
"""

import argparse
import http.client
import json
import logging
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import sklearn

from triagem.classificador import Classificador
from triagem.config import Settings
from triagem.corpus import carregar_corpus
from triagem.dados import baixar_arquivo

logger = logging.getLogger(__name__)

ARQUIVO_TESTE = "medical_tc_test.csv"
AMOSTRAS_PADRAO = 200
AQUECIMENTO_PADRAO = 100
MEDICOES_PADRAO = 1000
SEMENTE_PADRAO = 42
TIMEOUT_HTTP_SEGUNDOS = 10.0
ROTULO_PADRAO = "sklearn"


class RespostaHttpInvalidaError(RuntimeError):
    """A API respondeu com status diferente de 200 durante o benchmark."""


@dataclass(frozen=True)
class Parametros:
    """Parametros da medicao, registrados no relatorio."""

    amostras: int = AMOSTRAS_PADRAO
    aquecimento: int = AQUECIMENTO_PADRAO
    medicoes: int = MEDICOES_PADRAO
    semente: int = SEMENTE_PADRAO


@dataclass(frozen=True)
class Estatisticas:
    """Resumo de uma serie de latencias, em milissegundos."""

    medicoes: int
    media_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    maximo_ms: float


def calcular_estatisticas(latencias_ms: Sequence[float]) -> Estatisticas:
    """Resume as latencias. Percentis por interpolacao linear (padrao do numpy).

    Raises:
        ValueError: se a serie estiver vazia.
    """
    if len(latencias_ms) == 0:
        raise ValueError("Nenhuma latencia para resumir.")
    valores = np.asarray(latencias_ms, dtype=float)
    p50, p95, p99 = np.percentile(valores, [50, 95, 99])
    return Estatisticas(
        medicoes=len(valores),
        media_ms=round(float(valores.mean()), 3),
        p50_ms=round(float(p50), 3),
        p95_ms=round(float(p95), 3),
        p99_ms=round(float(p99), 3),
        maximo_ms=round(float(valores.max()), 3),
    )


def amostrar_textos(textos: Sequence[str], quantidade: int, semente: int) -> list[str]:
    """Sorteia `quantidade` textos distintos, de forma repetivel para a mesma semente.

    Raises:
        ValueError: se `quantidade` for invalida ou maior que o numero de textos.
    """
    if not 0 < quantidade <= len(textos):
        raise ValueError(f"quantidade deve estar entre 1 e {len(textos)}, recebido {quantidade}.")
    indices = np.random.default_rng(semente).choice(len(textos), size=quantidade, replace=False)
    return [textos[i] for i in indices]


def medir(
    chamada: Callable[[str], object],
    textos: Sequence[str],
    aquecimento: int,
    medicoes: int,
) -> list[float]:
    """Executa `aquecimento` chamadas descartadas e `medicoes` cronometradas, em ms.

    Os textos sao percorridos em ciclo, na ordem recebida.

    Raises:
        ValueError: sem textos, `aquecimento` negativo ou `medicoes` menor que 1.
    """
    if not textos:
        raise ValueError("Informe ao menos um texto.")
    if aquecimento < 0 or medicoes < 1:
        raise ValueError("aquecimento deve ser >= 0 e medicoes >= 1.")
    for i in range(aquecimento):
        chamada(textos[i % len(textos)])
    latencias: list[float] = []
    for i in range(medicoes):
        texto = textos[i % len(textos)]
        inicio = time.perf_counter()
        chamada(texto)
        latencias.append((time.perf_counter() - inicio) * 1000)
    return latencias


class ClienteHttp:
    """Cliente minimo da API, com conexao persistente (biblioteca padrao, sem dependencias)."""

    def __init__(self, url_base: str, timeout: float = TIMEOUT_HTTP_SEGUNDOS) -> None:
        partes = urlsplit(url_base)
        if partes.scheme != "http" or not partes.hostname:
            raise ValueError(f"Use uma URL no formato http://host:porta, recebido {url_base!r}.")
        self._conexao = http.client.HTTPConnection(
            partes.hostname, partes.port or 80, timeout=timeout
        )

    def versao_modelo(self) -> str:
        """Le a versao do modelo em `GET /health`."""
        self._conexao.request("GET", "/health")
        resposta = self._conexao.getresponse()
        corpo = resposta.read()
        if resposta.status != 200:
            raise RespostaHttpInvalidaError(f"/health respondeu {resposta.status}.")
        return str(json.loads(corpo)["versao_modelo"])

    def classificar(self, texto: str) -> None:
        """Envia `POST /predict` e le a resposta inteira (descartando o conteudo)."""
        corpo = json.dumps({"texto": texto}).encode("utf-8")
        self._conexao.request(
            "POST", "/predict", body=corpo, headers={"Content-Type": "application/json"}
        )
        resposta = self._conexao.getresponse()
        resposta.read()
        if resposta.status != 200:
            raise RespostaHttpInvalidaError(f"/predict respondeu {resposta.status}.")

    def fechar(self) -> None:
        """Fecha a conexao."""
        self._conexao.close()


def montar_relatorio(
    rotulo: str,
    parametros: Parametros,
    versao_modelo: str,
    resultados: dict[str, Estatisticas],
    versao_modelo_http: str | None = None,
) -> dict[str, Any]:
    """Monta o relatorio JSON com resultados, parametros e dados do ambiente."""
    return {
        "rotulo": rotulo,
        "gerado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "parametros": asdict(parametros),
        "modelo": {"versao_local": versao_modelo, "versao_http": versao_modelo_http},
        "ambiente": {
            "python": platform.python_version(),
            "plataforma": platform.platform(),
            "cpus": os.cpu_count(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "resultados": {nome: asdict(estatisticas) for nome, estatisticas in resultados.items()},
    }


def formatar_tabela(resultados: dict[str, Estatisticas]) -> str:
    """Tabela de texto simples para exibir no terminal."""
    linhas = [f"{'nivel':<12}{'media':>9}{'P50':>9}{'P95':>9}{'P99':>9}{'max':>9}  (ms)"]
    for nome, e in resultados.items():
        linhas.append(
            f"{nome:<12}{e.media_ms:>9.2f}{e.p50_ms:>9.2f}{e.p95_ms:>9.2f}"
            f"{e.p99_ms:>9.2f}{e.maximo_ms:>9.2f}"
        )
    return "\n".join(linhas)


def carregar_textos(settings: Settings, parametros: Parametros) -> list[str]:
    """Le o conjunto de teste (baixando e validando se preciso) e sorteia os laudos."""
    caminho = baixar_arquivo(ARQUIVO_TESTE, settings.raw_dir)
    return amostrar_textos(carregar_corpus(caminho).textos, parametros.amostras, parametros.semente)


def executar(
    classificador: Classificador,
    textos: Sequence[str],
    url: str | None,
    rotulo: str,
    parametros: Parametros,
) -> dict[str, Any]:
    """Roda o benchmark de inferencia pura e, se `url` for informada, o de HTTP.

    A API e consultada primeiro: se estiver fora do ar, falha antes de gastar tempo medindo.
    """
    cliente = ClienteHttp(url) if url else None
    versao_http: str | None = None
    resultados: dict[str, Estatisticas] = {}
    try:
        if cliente:
            versao_http = cliente.versao_modelo()
            if versao_http != classificador.versao:
                logger.warning(
                    "Modelo da API (%s) difere do local (%s): a comparacao entre niveis "
                    "nao e exata.",
                    versao_http,
                    classificador.versao,
                )
        logger.info(
            "Medindo inferencia pura (%d + %d chamadas)...",
            parametros.aquecimento,
            parametros.medicoes,
        )
        resultados["inferencia"] = calcular_estatisticas(
            medir(classificador.classificar, textos, parametros.aquecimento, parametros.medicoes)
        )
        if cliente:
            logger.info(
                "Medindo HTTP em %s (%d + %d chamadas)...",
                url,
                parametros.aquecimento,
                parametros.medicoes,
            )
            resultados["http"] = calcular_estatisticas(
                medir(cliente.classificar, textos, parametros.aquecimento, parametros.medicoes)
            )
    finally:
        if cliente:
            cliente.fechar()

    return montar_relatorio(rotulo, parametros, classificador.versao, resultados, versao_http)


def main(argv: Sequence[str] | None = None) -> None:
    """Ponto de entrada: python -m triagem.benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark de latencia da classificacao.")
    parser.add_argument(
        "--url", help="URL da API (ex.: http://localhost:8010); sem ela, so o modelo"
    )
    parser.add_argument("--rotulo", default=ROTULO_PADRAO, help="nome da variante medida")
    parser.add_argument("--amostras", type=int, default=AMOSTRAS_PADRAO)
    parser.add_argument("--aquecimento", type=int, default=AQUECIMENTO_PADRAO)
    parser.add_argument("--medicoes", type=int, default=MEDICOES_PADRAO)
    parser.add_argument("--semente", type=int, default=SEMENTE_PADRAO)
    parser.add_argument(
        "--saida", type=Path, help="JSON de saida (padrao: docs/benchmark_<rotulo>.json)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parametros = Parametros(args.amostras, args.aquecimento, args.medicoes, args.semente)
    settings = Settings.from_env()
    try:
        textos = carregar_textos(settings, parametros)
        classificador = Classificador.carregar(settings)
        relatorio = executar(classificador, textos, args.url, args.rotulo, parametros)
    except (OSError, ValueError, RuntimeError, http.client.HTTPException) as erro:
        logger.error("Benchmark falhou: %s", erro)
        sys.exit(1)

    resultados = {nome: Estatisticas(**dados) for nome, dados in relatorio["resultados"].items()}
    print(formatar_tabela(resultados))
    saida = args.saida or Path("docs") / f"benchmark_{args.rotulo}.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Relatorio gravado em %s", saida)


if __name__ == "__main__":
    main()
