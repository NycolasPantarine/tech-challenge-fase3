"""Gerador de carga para alimentar o dashboard de monitoramento.

Envia `POST /predict` a uma taxa constante, com laudos reais do conjunto de teste e uma
fracao de requisicoes invalidas (texto vazio ou sem conteudo), para que o painel de erros
mostre algo alem de zero. As requisicoes sao sequenciais numa unica conexao: serve para
gerar trafego visivel nos graficos, nao para medir capacidade (o benchmark cuida da latencia).

Os laudos vao so para a API; nada e gravado em disco nem impresso.

Uso: python -m triagem.carga [--url http://localhost:8000] [--duracao 120] [--taxa 10]
"""

import argparse
import logging
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from triagem.benchmark import ClienteHttp, Parametros, carregar_textos
from triagem.config import Settings

logger = logging.getLogger(__name__)

URL_PADRAO = "http://localhost:8000"
DURACAO_PADRAO = 120.0
TAXA_PADRAO = 10.0
INVALIDAS_PADRAO = 0.1
AMOSTRAS_PADRAO = 200
SEMENTE_PADRAO = 42
INTERVALO_PROGRESSO_SEGUNDOS = 10.0

# Ambos resultam em 422: o primeiro e barrado na validacao da entrada, o segundo so tem
# palavras sem conteudo e e recusado pelo classificador.
TEXTOS_INVALIDOS = ("", "!!! the of and ...")


@dataclass
class ResumoCarga:
    """Resultado de uma execucao: requisicoes por status HTTP e falhas de conexao."""

    por_status: Counter[int] = field(default_factory=Counter)
    falhas_conexao: int = 0
    interrompida: bool = False

    @property
    def enviadas(self) -> int:
        """Requisicoes que receberam resposta."""
        return sum(self.por_status.values())


def escolher_texto(
    sorteio: np.random.Generator, textos: Sequence[str], proporcao_invalidas: float
) -> str:
    """Sorteia o proximo texto: invalido com a probabilidade dada, senao um laudo real."""
    if sorteio.random() < proporcao_invalidas:
        return TEXTOS_INVALIDOS[int(sorteio.integers(len(TEXTOS_INVALIDOS)))]
    return textos[int(sorteio.integers(len(textos)))]


def executar(
    enviar: Callable[[str], int],
    textos: Sequence[str],
    duracao: float,
    taxa: float,
    proporcao_invalidas: float,
    semente: int,
    relogio: Callable[[], float] = time.monotonic,
    dormir: Callable[[float], None] = time.sleep,
) -> ResumoCarga:
    """Envia requisicoes a `taxa` por segundo durante `duracao` segundos.

    `enviar` recebe o texto e devolve o status HTTP; `OSError` conta como falha de conexao e a
    execucao continua. Ctrl+C encerra o laco e devolve o que foi contado ate ali.

    Raises:
        ValueError: parametros fora da faixa valida ou lista de textos vazia.
    """
    if not textos:
        raise ValueError("Informe ao menos um texto.")
    if duracao <= 0 or taxa <= 0:
        raise ValueError("duracao e taxa devem ser maiores que zero.")
    if not 0 <= proporcao_invalidas <= 1:
        raise ValueError("A proporcao de invalidas deve estar entre 0 e 1.")

    sorteio = np.random.default_rng(semente)
    resumo = ResumoCarga()
    intervalo = 1 / taxa
    inicio = relogio()
    proximo_envio = inicio
    ultimo_progresso = inicio
    try:
        while relogio() - inicio < duracao:
            try:
                status = enviar(escolher_texto(sorteio, textos, proporcao_invalidas))
            except OSError:
                resumo.falhas_conexao += 1
            else:
                resumo.por_status[status] += 1
            proximo_envio += intervalo
            espera = proximo_envio - relogio()
            if espera > 0:
                dormir(espera)
            if relogio() - ultimo_progresso >= INTERVALO_PROGRESSO_SEGUNDOS:
                ultimo_progresso = relogio()
                logger.info("%s", formatar_resumo(resumo))
    except KeyboardInterrupt:
        resumo.interrompida = True
    return resumo


def formatar_resumo(resumo: ResumoCarga) -> str:
    """Resumo em uma linha: total, status e falhas de conexao."""
    status = ", ".join(f"{codigo}: {qtd}" for codigo, qtd in sorted(resumo.por_status.items()))
    return f"enviadas={resumo.enviadas} [{status or '-'}] falhas_de_conexao={resumo.falhas_conexao}"


def main(argv: Sequence[str] | None = None) -> None:
    """Ponto de entrada: python -m triagem.carga."""
    parser = argparse.ArgumentParser(description="Gera trafego para a API de triagem.")
    parser.add_argument("--url", default=URL_PADRAO, help="URL da API")
    parser.add_argument("--duracao", type=float, default=DURACAO_PADRAO, help="segundos")
    parser.add_argument("--taxa", type=float, default=TAXA_PADRAO, help="requisicoes por segundo")
    parser.add_argument(
        "--invalidas",
        type=float,
        default=INVALIDAS_PADRAO,
        help="fracao de requisicoes invalidas, de 0 a 1",
    )
    parser.add_argument("--amostras", type=int, default=AMOSTRAS_PADRAO)
    parser.add_argument("--semente", type=int, default=SEMENTE_PADRAO)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cliente = ClienteHttp(args.url)

    def enviar(texto: str) -> int:
        try:
            return cliente.predizer(texto)
        except OSError:
            cliente.fechar()  # a proxima requisicao abre uma conexao nova
            raise

    try:
        textos = carregar_textos(Settings.from_env(), Parametros(amostras=args.amostras))
        logger.info("API em %s, modelo %s", args.url, cliente.versao_modelo())
        logger.info(
            "Enviando %.1f req/s por %.0f s (%.0f%% invalidas). Ctrl+C interrompe.",
            args.taxa,
            args.duracao,
            args.invalidas * 100,
        )
        resumo = executar(enviar, textos, args.duracao, args.taxa, args.invalidas, args.semente)
    except (OSError, ValueError, RuntimeError) as erro:
        logger.error("Carga falhou: %s", erro)
        sys.exit(1)
    finally:
        cliente.fechar()

    logger.info(
        "%s %s", "Interrompida." if resumo.interrompida else "Concluida.", formatar_resumo(resumo)
    )


if __name__ == "__main__":
    main()
