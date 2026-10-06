import json
from pathlib import Path

import pytest

from triagem.classificador import (
    Classificador,
    ModeloIndisponivelError,
    TextoSemConteudoError,
)
from triagem.config import Settings
from triagem.urgencia import Urgencia


@pytest.fixture(scope="module")
def classificador(settings_com_modelo: Settings) -> Classificador:
    return Classificador.carregar(settings_com_modelo)


def test_laudo_cardiovascular_e_urgente(classificador: Classificador) -> None:
    resultado = classificador.classificar("Cardiac coronary infarction with aortic involvement")

    assert resultado.urgencia is Urgencia.URGENTE
    assert resultado.categoria_id == 4
    assert resultado.categoria_nome == "cardiovascular diseases"


def test_laudo_de_neoplasia_e_atencao(classificador: Classificador) -> None:
    resultado = classificador.classificar("Tumor carcinoma with metastasis, oncology review")

    assert resultado.urgencia is Urgencia.ATENCAO
    assert resultado.categoria_id == 1


def test_probabilidades_de_urgencia_somam_um(classificador: Classificador) -> None:
    resultado = classificador.classificar("Gastric hepatic bowel disease")

    assert sum(resultado.probabilidades_urgencia.values()) == pytest.approx(1.0)
    assert set(resultado.probabilidades_urgencia) == set(Urgencia)


def test_texto_sem_termos_uteis_levanta_erro(classificador: Classificador) -> None:
    with pytest.raises(TextoSemConteudoError):
        classificador.classificar("the of and !!! ...")


def test_versao_identifica_modelo_e_data_do_treino(classificador: Classificador) -> None:
    assert classificador.versao == "naive_bayes_calibrado@2026-01-01T00:00:00+00:00"
    assert classificador.limiar_urgente == 0.5


def test_carregar_sem_artefato_levanta_erro(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path / "vazio")

    with pytest.raises(ModeloIndisponivelError, match="python -m triagem.treino"):
        Classificador.carregar(settings)


def test_carregar_com_versao_diferente_do_scikit_learn_levanta_erro(
    settings_com_modelo: Settings, tmp_path: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "modelo.joblib").write_bytes(settings_com_modelo.model_path.read_bytes())
    metadata = json.loads(settings_com_modelo.metadata_path.read_text(encoding="utf-8"))
    metadata["versoes"]["scikit_learn"] = "0.0.1"
    (artifacts / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ModeloIndisponivelError, match="0.0.1"):
        Classificador.carregar(Settings(data_dir=tmp_path, artifacts_dir=artifacts))


def test_carregar_com_metadados_ilegiveis_levanta_erro(
    settings_com_modelo: Settings, tmp_path: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "modelo.joblib").write_bytes(settings_com_modelo.model_path.read_bytes())
    (artifacts / "metadata.json").write_text("{ isto nao e json", encoding="utf-8")

    with pytest.raises(ModeloIndisponivelError, match="ilegiveis"):
        Classificador.carregar(Settings(data_dir=tmp_path, artifacts_dir=artifacts))
