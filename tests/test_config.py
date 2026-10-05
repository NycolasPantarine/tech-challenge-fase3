from pathlib import Path

import pytest

from triagem.config import ENV_ARTIFACTS_DIR, ENV_DATA_DIR, Settings


def test_padroes_sem_variaveis_de_ambiente(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_DATA_DIR, raising=False)
    monkeypatch.delenv(ENV_ARTIFACTS_DIR, raising=False)

    settings = Settings.from_env()

    assert settings.data_dir == Path("data")
    assert settings.artifacts_dir == Path("artifacts")
    assert settings.raw_dir == Path("data") / "raw"


def test_variaveis_de_ambiente_sobrescrevem_padroes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "dados"))
    monkeypatch.setenv(ENV_ARTIFACTS_DIR, str(tmp_path / "modelos"))

    settings = Settings.from_env()

    assert settings.data_dir == tmp_path / "dados"
    assert settings.artifacts_dir == tmp_path / "modelos"


def test_settings_e_imutavel() -> None:
    settings = Settings(data_dir=Path("a"), artifacts_dir=Path("b"))

    with pytest.raises(AttributeError):
        settings.data_dir = Path("c")


def test_caminhos_dos_artefatos(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path / "artifacts")

    assert settings.model_path == tmp_path / "artifacts" / "modelo.joblib"
    assert settings.metadata_path == tmp_path / "artifacts" / "metadata.json"
