import pytest

from triagem.modelo import TipoModelo, construir_pipeline

TEXTOS = ["tumor carcinoma"] * 6 + ["cardiac coronary"] * 6 + ["neuron seizure"] * 6
CATEGORIAS = [1] * 6 + [4] * 6 + [3] * 6


@pytest.mark.parametrize("tipo", list(TipoModelo))
def test_todo_candidato_treina_e_gera_probabilidades(tipo: TipoModelo) -> None:
    pipeline = construir_pipeline(tipo).fit(TEXTOS, CATEGORIAS)

    probabilidades = pipeline.predict_proba(["cardiac coronary"])

    assert probabilidades.shape == (1, 3)
    assert probabilidades.sum() == pytest.approx(1.0)
    assert pipeline.predict(["cardiac coronary"])[0] == 4


def test_naive_bayes_calibrado_difere_do_simples() -> None:
    simples = construir_pipeline(TipoModelo.NAIVE_BAYES_SIMPLES)
    calibrado = construir_pipeline(TipoModelo.NAIVE_BAYES)

    assert type(simples.named_steps["clf"]) is not type(calibrado.named_steps["clf"])
