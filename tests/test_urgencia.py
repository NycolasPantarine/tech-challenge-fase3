import pytest

from triagem.urgencia import (
    CATEGORIA_PARA_URGENCIA,
    CATEGORIAS,
    Urgencia,
    agregar_probabilidades,
    urgencia_da_categoria,
)


def test_toda_categoria_do_dataset_tem_urgencia() -> None:
    assert set(CATEGORIA_PARA_URGENCIA) == set(CATEGORIAS)


def test_todos_os_niveis_de_urgencia_sao_usados() -> None:
    assert set(CATEGORIA_PARA_URGENCIA.values()) == set(Urgencia)


@pytest.mark.parametrize(
    ("categoria", "esperada"),
    [
        (1, Urgencia.ATENCAO),
        (2, Urgencia.ATENCAO),
        (3, Urgencia.URGENTE),
        (4, Urgencia.URGENTE),
        (5, Urgencia.NORMAL),
    ],
)
def test_urgencia_da_categoria(categoria: int, esperada: Urgencia) -> None:
    assert urgencia_da_categoria(categoria) is esperada


def test_urgencia_de_categoria_desconhecida_levanta_erro() -> None:
    with pytest.raises(ValueError, match="desconhecida"):
        urgencia_da_categoria(99)


def test_agregar_probabilidades_soma_por_nivel() -> None:
    probabilidades = {1: 0.10, 2: 0.10, 3: 0.20, 4: 0.30, 5: 0.30}

    agregado = agregar_probabilidades(probabilidades)

    assert agregado[Urgencia.URGENTE] == pytest.approx(0.50)
    assert agregado[Urgencia.ATENCAO] == pytest.approx(0.20)
    assert agregado[Urgencia.NORMAL] == pytest.approx(0.30)
    assert sum(agregado.values()) == pytest.approx(1.0)


def test_agregar_probabilidades_exige_todas_as_categorias() -> None:
    with pytest.raises(ValueError, match="Esperadas"):
        agregar_probabilidades({1: 0.5, 2: 0.5})
