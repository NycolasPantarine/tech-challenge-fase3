"""Fixtures compartilhadas: corpus sintetico e artefato de modelo pequeno."""

import json
from collections.abc import Callable

import joblib
import pytest
import sklearn

from triagem.config import Settings
from triagem.corpus import Corpus
from triagem.modelo import TipoModelo, construir_pipeline
from triagem.texto import normalizar

# Vocabulario exclusivo por categoria, para o problema ser separavel e o teste, deterministico.
VOCABULARIO = {
    1: "tumor carcinoma neoplasm metastasis oncology",
    2: "gastric hepatic colon bowel pancreas",
    3: "neuron cerebral seizure dementia stroke",
    4: "cardiac aortic coronary ventricular infarction",
    5: "infection fever inflammation sepsis syndrome",
}


def _criar_corpus(repeticoes: int) -> Corpus:
    textos, categorias = [], []
    for categoria, palavras in VOCABULARIO.items():
        for i in range(repeticoes):
            textos.append(f"{palavras} {palavras} caso {i}")
            categorias.append(categoria)
    return Corpus(textos=textos, categorias=categorias)


@pytest.fixture
def fabrica_corpus() -> Callable[[int], Corpus]:
    """Cria um corpus sintetico com N documentos por categoria."""
    return _criar_corpus


@pytest.fixture(scope="session")
def settings_com_modelo(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Artefato real (pipeline treinado + metadados) em diretorio temporario."""
    base = tmp_path_factory.mktemp("modelo")
    settings = Settings(data_dir=base / "data", artifacts_dir=base / "artifacts")
    corpus = _criar_corpus(30)
    pipeline = construir_pipeline(TipoModelo.NAIVE_BAYES)
    pipeline.fit([normalizar(t) for t in corpus.textos], corpus.categorias)

    settings.artifacts_dir.mkdir(parents=True)
    joblib.dump(pipeline, settings.model_path)
    metadata = {
        "tipo_modelo": TipoModelo.NAIVE_BAYES.value,
        "treinado_em": "2026-01-01T00:00:00+00:00",
        "versoes": {"scikit_learn": sklearn.__version__},
        "limiar_urgente": 0.5,
    }
    settings.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return settings
