import pytest

from triagem.urgencia import (
    CATEGORIA_PARA_URGENCIA,
    CATEGORIAS,
    Urgencia,
    agregar_probabilidades,
    decidir_urgencia,
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


def test_decidir_urgencia_sem_limiar_usa_maior_probabilidade() -> None:
    probabilidades = {Urgencia.NORMAL: 0.2, Urgencia.ATENCAO: 0.5, Urgencia.URGENTE: 0.3}

    assert decidir_urgencia(probabilidades) is Urgencia.ATENCAO


def test_decidir_urgencia_com_limiar_prioriza_urgente() -> None:
    probabilidades = {Urgencia.NORMAL: 0.2, Urgencia.ATENCAO: 0.5, Urgencia.URGENTE: 0.3}

    assert decidir_urgencia(probabilidades, limiar_urgente=0.3) is Urgencia.URGENTE
    assert decidir_urgencia(probabilidades, limiar_urgente=0.31) is Urgencia.ATENCAO


def test_decidir_urgencia_com_limiar_alto_escolhe_entre_normal_e_atencao() -> None:
    probabilidades = {Urgencia.NORMAL: 0.6, Urgencia.ATENCAO: 0.1, Urgencia.URGENTE: 0.3}

    assert decidir_urgencia(probabilidades, limiar_urgente=0.9) is Urgencia.NORMAL


@pytest.mark.parametrize("limiar", [-0.1, 1.1])
def test_decidir_urgencia_rejeita_limiar_invalido(limiar: float) -> None:
    probabilidades = {Urgencia.NORMAL: 0.4, Urgencia.ATENCAO: 0.3, Urgencia.URGENTE: 0.3}

    with pytest.raises(ValueError, match="Limiar"):
        decidir_urgencia(probabilidades, limiar_urgente=limiar)
