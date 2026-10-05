"""Download verificado do Medical Abstracts TC Corpus.

Os dados nao ficam no git (licenca CC BY-SA 3.0, arquivos grandes): sao baixados por este
modulo e validados por SHA-256, o que garante a mesma entrada em qualquer maquina.

Fonte: https://github.com/sebischair/Medical-Abstracts-TC-Corpus
"""

import hashlib
import logging
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from triagem.config import Settings

logger = logging.getLogger(__name__)

URL_BASE = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main"
TIMEOUT_SEGUNDOS = 60

# nome do arquivo -> SHA-256 esperado
ARQUIVOS: dict[str, str] = {
    "medical_tc_train.csv": "ad53aebc682d6b87a5647f619a079bb446d286fdc93bf0159b812418f5758609",
    "medical_tc_test.csv": "1eecea73c9ecad292c55e10403bd139fab9580545d6878482997c5564d51ac05",
    "medical_tc_labels.csv": "8a27ae03339c798103678efa8012f744a723ff71a80f2b2c1355ee249564adc5",
}

Fetcher = Callable[[str], bytes]


class IntegridadeError(RuntimeError):
    """O conteudo baixado nao corresponde ao hash esperado."""


def baixar_https(url: str) -> bytes:
    """Baixa uma URL https. Recusa qualquer outro esquema (ex.: file://)."""
    if urlparse(url).scheme != "https":
        raise ValueError(f"Somente https e permitido: {url!r}")
    with urllib.request.urlopen(url, timeout=TIMEOUT_SEGUNDOS) as resposta:  # noqa: S310
        return resposta.read()


def _sha256(conteudo: bytes) -> str:
    return hashlib.sha256(conteudo).hexdigest()


def _hash_do_arquivo(caminho: Path) -> str:
    return _sha256(caminho.read_bytes())


def _gravar_atomicamente(destino: Path, conteudo: bytes) -> None:
    """Grava via arquivo temporario no mesmo diretorio, para nunca deixar arquivo parcial."""
    with tempfile.NamedTemporaryFile(dir=destino.parent, delete=False) as tmp:
        tmp.write(conteudo)
        caminho_tmp = Path(tmp.name)
    caminho_tmp.replace(destino)


def baixar_arquivo(nome: str, destino_dir: Path, fetch: Fetcher = baixar_https) -> Path:
    """Garante que o arquivo exista em destino_dir com o hash esperado.

    Se ja existir e estiver integro, nao baixa de novo. Se existir corrompido, baixa outra vez.

    Raises:
        KeyError: se o arquivo nao estiver no catalogo.
        IntegridadeError: se o conteudo baixado nao bater com o SHA-256 esperado.
    """
    esperado = ARQUIVOS[nome]
    destino_dir.mkdir(parents=True, exist_ok=True)
    caminho = destino_dir / nome

    if caminho.exists() and _hash_do_arquivo(caminho) == esperado:
        logger.info("%s ja existe e esta integro.", nome)
        return caminho

    logger.info("Baixando %s ...", nome)
    conteudo = fetch(f"{URL_BASE}/{nome}")
    obtido = _sha256(conteudo)
    if obtido != esperado:
        raise IntegridadeError(f"{nome}: SHA-256 esperado {esperado}, obtido {obtido}")
    _gravar_atomicamente(caminho, conteudo)
    return caminho


def baixar_dataset(settings: Settings, fetch: Fetcher = baixar_https) -> dict[str, Path]:
    """Baixa e valida todos os arquivos do dataset em settings.raw_dir."""
    return {nome: baixar_arquivo(nome, settings.raw_dir, fetch) for nome in ARQUIVOS}


def main() -> None:
    """Ponto de entrada: python -m triagem.dados."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    caminhos = baixar_dataset(Settings.from_env())
    for nome, caminho in caminhos.items():
        logger.info("OK %s -> %s", nome, caminho)


if __name__ == "__main__":
    main()
