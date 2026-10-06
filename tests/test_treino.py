import json
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pytest

from triagem.config import Settings
from triagem.corpus import Corpus
from triagem.modelo import TipoModelo, construir_pipeline, preparar_para_inferencia
from triagem.texto import normalizar
from triagem.treino import ResultadoCandidato, executar_treino, selecionar_melhor


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")


def test_fluxo_completo_gera_modelo_e_metadados(
    settings: Settings, fabrica_corpus: Callable[[int], Corpus]
) -> None:
    metadata = executar_treino(
        settings,
        fabrica_corpus(30),
        fabrica_corpus(10),
        folds=2,
        recall_alvo=0.8,
        hash_treino="abc",
    )

    assert settings.model_path.exists()
    assert (
        json.loads(settings.metadata_path.read_text(encoding="utf-8"))["sha256_dados_treino"]
        == "abc"
    )
    assert set(metadata["candidatos_cv"]) == {t.value for t in TipoModelo}
    assert metadata["teste_com_limiar"]["recall_urgente"] >= 0.8
    assert metadata["classes"] == [1, 2, 3, 4, 5]


def test_modelo_salvo_preve_categoria_correta(
    settings: Settings, fabrica_corpus: Callable[[int], Corpus]
) -> None:
    executar_treino(settings, fabrica_corpus(30), fabrica_corpus(10), folds=2)

    pipeline = joblib.load(settings.model_path)
    texto = normalizar("Cardiac coronary infarction in aortic ventricular disease")

    assert pipeline.predict([texto])[0] == 4


def test_floresta_salva_para_inferencia_usa_thread_unica() -> None:
    pipeline = construir_pipeline(TipoModelo.FLORESTA, n_jobs=-1)

    preparar_para_inferencia(pipeline)

    assert pipeline.get_params()["clf__n_jobs"] == 1


def test_selecionar_melhor_usa_f1_macro_de_urgencia() -> None:
    vazio = np.zeros((1, 5))
    fraco = ResultadoCandidato(TipoModelo.NAIVE_BAYES, {"f1_macro_urgencia": 0.4}, vazio, 1.0)
    forte = ResultadoCandidato(TipoModelo.LOGISTICA, {"f1_macro_urgencia": 0.6}, vazio, 1.0)

    assert selecionar_melhor([fraco, forte]).tipo is TipoModelo.LOGISTICA
