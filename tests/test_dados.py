import hashlib
from pathlib import Path

import pytest

from triagem import dados
from triagem.config import Settings
from triagem.dados import IntegridadeError, baixar_arquivo, baixar_dataset, baixar_https

CONTEUDO = b"condition_label,medical_abstract\n1,texto de teste\n"
NOME = "medical_tc_train.csv"


@pytest.fixture
def hash_falso(monkeypatch: pytest.MonkeyPatch) -> str:
    """Troca o hash esperado do arquivo de teste pelo hash do CONTEUDO de teste."""
    esperado = hashlib.sha256(CONTEUDO).hexdigest()
    monkeypatch.setitem(dados.ARQUIVOS, NOME, esperado)
    return esperado


def test_baixa_e_grava_arquivo_integro(tmp_path: Path, hash_falso: str) -> None:
    caminho = baixar_arquivo(NOME, tmp_path, fetch=lambda _url: CONTEUDO)

    assert caminho.read_bytes() == CONTEUDO


def test_hash_divergente_levanta_erro_e_nao_deixa_arquivo(tmp_path: Path, hash_falso: str) -> None:
    with pytest.raises(IntegridadeError):
        baixar_arquivo(NOME, tmp_path, fetch=lambda _url: b"conteudo adulterado")

    assert list(tmp_path.iterdir()) == []


def test_nao_baixa_de_novo_se_arquivo_ja_esta_integro(tmp_path: Path, hash_falso: str) -> None:
    (tmp_path / NOME).write_bytes(CONTEUDO)

    def fetch_proibido(_url: str) -> bytes:
        raise AssertionError("nao deveria baixar")

    baixar_arquivo(NOME, tmp_path, fetch=fetch_proibido)


def test_rebaixa_se_arquivo_local_estiver_corrompido(tmp_path: Path, hash_falso: str) -> None:
    (tmp_path / NOME).write_bytes(b"corrompido")

    caminho = baixar_arquivo(NOME, tmp_path, fetch=lambda _url: CONTEUDO)

    assert caminho.read_bytes() == CONTEUDO


def test_arquivo_fora_do_catalogo_levanta_keyerror(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        baixar_arquivo("nao_existe.csv", tmp_path, fetch=lambda _url: b"")


def test_baixar_dataset_baixa_todos_os_arquivos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        dados, "ARQUIVOS", {n: hashlib.sha256(CONTEUDO).hexdigest() for n in dados.ARQUIVOS}
    )
    settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path / "artifacts")

    caminhos = baixar_dataset(settings, fetch=lambda _url: CONTEUDO)

    assert set(caminhos) == set(dados.ARQUIVOS)
    assert all(caminho.exists() for caminho in caminhos.values())


@pytest.mark.parametrize("url", ["http://exemplo.com/a.csv", "file:///etc/passwd", "ftp://x/y"])
def test_baixar_https_recusa_outros_esquemas(url: str) -> None:
    with pytest.raises(ValueError, match="https"):
        baixar_https(url)
