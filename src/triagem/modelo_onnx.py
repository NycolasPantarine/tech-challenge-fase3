"""Inferencia com o modelo ONNX (so ONNX Runtime; nada de scikit-learn ou skl2onnx).

O arquivo e gerado por `triagem.onnx_export`. Esta classe e o que a API usa em execucao.

Tokens com caractere nao-ASCII: o tokenizador do ONNX corta o texto em `[A-Za-z0-9_]+`,
enquanto o do scikit-learn usa `\\w` (Unicode). Um token como `resume` com acentos viraria
dois pedacos no ONNX e uma unica palavra desconhecida no scikit-learn, e os pedacos podem
existir no vocabulario. Como o vocabulario do modelo e so ASCII (dataset em ingles), todo
token com caractere nao-ASCII e desconhecido para o scikit-learn; aqui ele e trocado por um
token reservado tambem desconhecido, o que da o mesmo resultado nos dois motores. A
exportacao confere que o token reservado nao esta no vocabulario.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

PALAVRA_RESERVADA = "tokenforadovocabulario"
ENTRADA = "texto"
SAIDA_PROBABILIDADES = "probabilities"  # nome gerado pelo skl2onnx


def preparar_texto(normalizado: str) -> str:
    """Troca tokens com caractere nao-ASCII pelo token reservado (ver docstring do modulo)."""
    if normalizado.isascii():
        return normalizado
    return " ".join(t if t.isascii() else PALAVRA_RESERVADA for t in normalizado.split())


class ModeloOnnx:
    """Sessao do ONNX Runtime que devolve a probabilidade de cada categoria."""

    def __init__(self, sessao: "InferenceSession", classes: Sequence[int]) -> None:
        self._sessao = sessao
        self.classes = [int(c) for c in classes]

    @classmethod
    def abrir(cls, caminho: Path, classes: Sequence[int]) -> "ModeloOnnx":
        """Abre o modelo em uma unica thread, como o modelo original (D-019).

        Raises:
            OSError, ValueError, RuntimeError: arquivo ausente, corrompido ou incompativel.
        """
        import onnxruntime as ort  # importado aqui: so quem usa o backend ONNX paga o custo

        opcoes = ort.SessionOptions()
        opcoes.intra_op_num_threads = 1
        opcoes.inter_op_num_threads = 1
        opcoes.log_severity_level = 3  # so erros: o grafo gera avisos inofensivos
        sessao = ort.InferenceSession(str(caminho), opcoes, providers=["CPUExecutionProvider"])
        saidas = {saida.name for saida in sessao.get_outputs()}
        if SAIDA_PROBABILIDADES not in saidas:
            raise ValueError(f"Modelo ONNX sem a saida {SAIDA_PROBABILIDADES!r} (tem {saidas}).")
        return cls(sessao, classes)

    def prever(self, normalizado: str) -> np.ndarray:
        """Probabilidade por categoria, na ordem de `classes`, para um texto ja normalizado."""
        entrada = np.array([[preparar_texto(normalizado)]], dtype=object)
        saida = self._sessao.run([SAIDA_PROBABILIDADES], {ENTRADA: entrada})[0]
        return np.asarray(saida[0], dtype=np.float64)
