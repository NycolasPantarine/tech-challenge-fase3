import json
from collections.abc import Callable
from pathlib import Path

import pytest

from triagem.classificador import Classificador, ModeloIndisponivelError
from triagem.config import Settings
from triagem.corpus import Corpus
from triagem.retreino import (
    PASTA_STAGING,
    ModeloReprovadoError,
    promover_staging,
    treinar_em_staging,
    validar_staging,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")


@pytest.fixture
def staging(settings: Settings, fabrica_corpus: Callable[[int], Corpus]) -> Path:
    return treinar_em_staging(settings, fabrica_corpus(30), fabrica_corpus(10), folds=2)


def test_treino_vai_para_staging_sem_tocar_no_modelo_publicado(staging: Path, settings: Settings):
    assert (staging / "modelo.joblib").exists()
    assert (staging / "metadata.json").exists()
    assert staging.parent == settings.artifacts_dir / PASTA_STAGING
    assert not settings.model_path.exists()
    assert not settings.metadata_path.exists()


def test_validar_aprova_modelo_bom(staging: Path, settings: Settings):
    resultado = validar_staging(settings, staging)

    assert resultado["f1_urgencia"] >= 0.55
    assert resultado["recall_urgente"] >= 0.75
    assert "@" in resultado["versao"]


def test_validar_reprova_metricas_abaixo_do_minimo(staging: Path, settings: Settings):
    with pytest.raises(ModeloReprovadoError, match="reprovado"):
        validar_staging(settings, staging, f1_minimo=1.1)


def test_validar_reprova_recall_abaixo_do_minimo(staging: Path, settings: Settings):
    with pytest.raises(ModeloReprovadoError, match="recall"):
        validar_staging(settings, staging, recall_minimo=1.1)


def test_validar_falha_se_artefato_ausente(settings: Settings):
    vazio = settings.artifacts_dir / PASTA_STAGING / "vazio"
    vazio.mkdir(parents=True)

    with pytest.raises(ModeloIndisponivelError):
        validar_staging(settings, vazio)


def test_promover_publica_o_modelo_e_remove_o_staging(staging: Path, settings: Settings):
    promover_staging(settings, staging)

    assert not staging.exists()
    assert Classificador.carregar(settings).classificar("cardiac infarction").urgencia


def test_promover_substitui_o_modelo_anterior(staging: Path, settings: Settings):
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"modelo antigo")
    settings.metadata_path.write_text("{}", encoding="utf-8")
    versao_nova = json.loads((staging / "metadata.json").read_text(encoding="utf-8"))

    promover_staging(settings, staging)

    metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    assert metadata["treinado_em"] == versao_nova["treinado_em"]
    assert settings.model_path.read_bytes() != b"modelo antigo"


def test_promover_recusa_diretorio_fora_do_staging(settings: Settings, tmp_path: Path):
    intruso = tmp_path / "outro_lugar"
    intruso.mkdir()
    (intruso / "modelo.joblib").write_bytes(b"x")

    with pytest.raises(ValueError, match="staging valido"):
        promover_staging(settings, intruso)

    assert (intruso / "modelo.joblib").exists()


def test_modelo_reprovado_nao_afeta_o_modelo_publicado(
    staging: Path, settings: Settings, fabrica_corpus: Callable[[int], Corpus]
):
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"modelo em uso")

    with pytest.raises(ModeloReprovadoError):
        validar_staging(settings, staging, f1_minimo=1.1)

    assert settings.model_path.read_bytes() == b"modelo em uso"
