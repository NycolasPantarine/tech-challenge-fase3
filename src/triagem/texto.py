"""Normalizacao de texto, aplicada identicamente no treino e na inferencia.

A normalizacao (minusculas, tokenizacao e remocao de stop words) fica FORA do pipeline do
scikit-learn de proposito: o TfidfVectorizer com `lowercase=True` ou `stop_words` vira um
operador ONNX (StringNormalizer) que exige locale do sistema e quebra em imagens slim.
Fazendo aqui, o vetorizador recebe tokens limpos e a conversao para ONNX fica portavel.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Mesmo padrao de token padrao do scikit-learn: 2+ caracteres alfanumericos.
_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def normalizar(texto: str) -> str:
    """Converte para minusculas, tokeniza e remove stop words em ingles.

    Retorna os tokens separados por um espaco. Texto sem tokens uteis vira string vazia.
    """
    tokens = _TOKEN.findall(texto.lower())
    return " ".join(token for token in tokens if token not in ENGLISH_STOP_WORDS)
