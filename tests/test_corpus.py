from pathlib import Path

import pytest

from triagem.corpus import CorpusInvalidoError, carregar_corpus

CABECALHO = "condition_label,medical_abstract\n"


def _escrever(tmp_path: Path, conteudo: str) -> Path:
    caminho = tmp_path / "corpus.csv"
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


def test_carrega_corpus_valido(tmp_path: Path) -> None:
    caminho = _escrever(tmp_path, CABECALHO + '1,"texto um, com virgula"\n4,texto dois\n')

    corpus = carregar_corpus(caminho)

    assert corpus.categorias == [1, 4]
    assert corpus.textos == ["texto um, com virgula", "texto dois"]
    assert len(corpus) == 2


def test_colunas_ausentes_levantam_erro(tmp_path: Path) -> None:
    caminho = _escrever(tmp_path, "a,b\n1,x\n")

    with pytest.raises(CorpusInvalidoError, match="colunas"):
        carregar_corpus(caminho)


def test_categoria_desconhecida_levanta_erro(tmp_path: Path) -> None:
    caminho = _escrever(tmp_path, CABECALHO + "9,texto\n")

    with pytest.raises(CorpusInvalidoError, match="desconhecida"):
        carregar_corpus(caminho)


def test_categoria_nao_numerica_levanta_erro(tmp_path: Path) -> None:
    caminho = _escrever(tmp_path, CABECALHO + "abc,texto\n")

    with pytest.raises(CorpusInvalidoError, match="invalida"):
        carregar_corpus(caminho)


def test_texto_vazio_levanta_erro(tmp_path: Path) -> None:
    caminho = _escrever(tmp_path, CABECALHO + "1,   \n")

    with pytest.raises(CorpusInvalidoError, match="vazio"):
        carregar_corpus(caminho)


def test_arquivo_sem_registros_levanta_erro(tmp_path: Path) -> None:
    caminho = _escrever(tmp_path, CABECALHO)

    with pytest.raises(CorpusInvalidoError, match="nenhum registro"):
        carregar_corpus(caminho)
