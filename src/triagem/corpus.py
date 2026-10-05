"""Leitura e validacao dos CSVs do Medical Abstracts TC Corpus."""

import csv
from dataclasses import dataclass
from pathlib import Path

from triagem.urgencia import CATEGORIAS

COLUNA_CATEGORIA = "condition_label"
COLUNA_TEXTO = "medical_abstract"


class CorpusInvalidoError(ValueError):
    """O arquivo nao segue o formato esperado do corpus."""


@dataclass(frozen=True)
class Corpus:
    """Textos e categorias alinhados por posicao."""

    textos: list[str]
    categorias: list[int]

    def __len__(self) -> int:
        return len(self.textos)


def carregar_corpus(caminho: Path) -> Corpus:
    """Le um CSV do corpus validando colunas, categorias e textos vazios.

    Raises:
        CorpusInvalidoError: colunas ausentes, categoria desconhecida, texto vazio ou arquivo vazio.
    """
    textos: list[str] = []
    categorias: list[int] = []
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        if not {COLUNA_CATEGORIA, COLUNA_TEXTO} <= set(leitor.fieldnames or []):
            raise CorpusInvalidoError(
                f"{caminho.name}: esperadas as colunas {COLUNA_CATEGORIA!r} e {COLUNA_TEXTO!r}"
            )
        for numero, linha in enumerate(leitor, start=2):
            try:
                categoria = int(linha[COLUNA_CATEGORIA])
            except ValueError:
                raise CorpusInvalidoError(
                    f"{caminho.name} linha {numero}: categoria invalida {linha[COLUNA_CATEGORIA]!r}"
                ) from None
            if categoria not in CATEGORIAS:
                raise CorpusInvalidoError(
                    f"{caminho.name} linha {numero}: categoria desconhecida {categoria}"
                )
            texto = linha[COLUNA_TEXTO].strip()
            if not texto:
                raise CorpusInvalidoError(f"{caminho.name} linha {numero}: texto vazio")
            textos.append(texto)
            categorias.append(categoria)
    if not textos:
        raise CorpusInvalidoError(f"{caminho.name}: nenhum registro")
    return Corpus(textos=textos, categorias=categorias)
