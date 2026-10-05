"""Mapeamento das categorias clinicas do dataset para niveis de urgencia.

O Medical Abstracts TC Corpus classifica cada resumo por categoria de doenca, nao por
urgencia. O mapeamento abaixo e uma HEURISTICA de dominio, nao um criterio clinico: ele
agrupa categorias conforme a tendencia a eventos agudos. Fica isolado neste modulo para
poder ser revisado sem tocar no treino, e a API expoe tambem a categoria original.
"""

from collections.abc import Mapping
from enum import StrEnum


class Urgencia(StrEnum):
    """Niveis de urgencia da triagem."""

    NORMAL = "normal"
    ATENCAO = "atencao"
    URGENTE = "urgente"


# Rotulos do dataset (condition_label) e seus nomes.
CATEGORIAS: Mapping[int, str] = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}

CATEGORIA_PARA_URGENCIA: Mapping[int, Urgencia] = {
    1: Urgencia.ATENCAO,
    2: Urgencia.ATENCAO,
    3: Urgencia.URGENTE,
    4: Urgencia.URGENTE,
    5: Urgencia.NORMAL,
}


def urgencia_da_categoria(categoria: int) -> Urgencia:
    """Retorna a urgencia associada a uma categoria do dataset.

    Raises:
        ValueError: se a categoria nao existir no dataset.
    """
    try:
        return CATEGORIA_PARA_URGENCIA[categoria]
    except KeyError:
        raise ValueError(f"Categoria desconhecida: {categoria!r}") from None


def agregar_probabilidades(probabilidades: Mapping[int, float]) -> dict[Urgencia, float]:
    """Soma as probabilidades das categorias que pertencem a cada nivel de urgencia.

    Args:
        probabilidades: probabilidade prevista por categoria (chaves 1 a 5).

    Returns:
        Probabilidade por nivel de urgencia, com todos os niveis presentes.

    Raises:
        ValueError: se faltar alguma categoria ou houver categoria desconhecida.
    """
    if set(probabilidades) != set(CATEGORIAS):
        raise ValueError(
            f"Esperadas as categorias {sorted(CATEGORIAS)}, recebidas {sorted(probabilidades)}"
        )
    agregado = dict.fromkeys(Urgencia, 0.0)
    for categoria, probabilidade in probabilidades.items():
        agregado[CATEGORIA_PARA_URGENCIA[categoria]] += probabilidade
    return agregado
