"""Garante que a imagem do Airflow treina com as mesmas versoes que a API usa."""

import tomllib
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def _pins_do_airflow() -> dict[str, str]:
    pins = {}
    for linha in (RAIZ / "airflow" / "requirements-treino.txt").read_text().splitlines():
        linha = linha.strip()
        if linha and not linha.startswith("#"):
            nome, versao = linha.split("==")
            pins[nome.lower()] = versao
    return pins


def test_versoes_do_airflow_batem_com_o_poetry_lock() -> None:
    lock = tomllib.loads((RAIZ / "poetry.lock").read_text(encoding="utf-8"))
    versoes_lock = {p["name"].lower(): p["version"] for p in lock["package"]}

    pins = _pins_do_airflow()

    assert pins, "requirements-treino.txt vazio"
    for nome, versao in pins.items():
        assert versoes_lock[nome] == versao, f"{nome}: lock={versoes_lock[nome]} airflow={versao}"
