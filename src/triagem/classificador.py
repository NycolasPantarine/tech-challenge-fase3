"""Servico de classificacao: carrega o artefato treinado e classifica laudos.

Nao depende do FastAPI, entao a logica pode ser testada sem subir servidor. O motor de
inferencia (pipeline do scikit-learn ou modelo ONNX) e escolhido por `Settings.backend` e
isolado atras de `ModeloDeProbabilidades`: o restante do servico nao sabe qual esta em uso.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

from triagem.config import Backend, Settings
from triagem.modelo_onnx import ModeloOnnx
from triagem.texto import normalizar
from triagem.urgencia import CATEGORIAS, Urgencia, agregar_probabilidades, decidir_urgencia

logger = logging.getLogger(__name__)

SUFIXO_VERSAO_ONNX = "+onnx"


class ModeloIndisponivelError(RuntimeError):
    """O artefato do modelo esta ausente, ilegivel ou incompativel com o ambiente."""


class TextoSemConteudoError(ValueError):
    """O texto nao tem nenhum termo util depois da normalizacao."""


class ModeloDeProbabilidades(Protocol):
    """Motor de inferencia: probabilidade por categoria para um texto ja normalizado."""

    classes: list[int]

    def prever(self, normalizado: str) -> np.ndarray:
        """Probabilidades na ordem de `classes`."""
        ...


class _ModeloSklearn:
    """Pipeline original do scikit-learn."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline
        self.classes = [int(c) for c in pipeline.classes_]

    def prever(self, normalizado: str) -> np.ndarray:
        return self._pipeline.predict_proba([normalizado])[0]


@dataclass(frozen=True)
class Resultado:
    """Resultado de uma classificacao."""

    urgencia: Urgencia
    probabilidades_urgencia: dict[Urgencia, float]
    categoria_id: int
    categoria_nome: str
    categoria_probabilidade: float


class Classificador:
    """Motor de inferencia treinado, limiar de urgencia e versao do modelo."""

    def __init__(self, modelo: ModeloDeProbabilidades, limiar_urgente: float, versao: str) -> None:
        self._modelo = modelo
        self.limiar_urgente = limiar_urgente
        self.versao = versao

    @classmethod
    def carregar(cls, settings: Settings) -> "Classificador":
        """Carrega o artefato gerado por `python -m triagem.treino`.

        Com `Backend.ONNX`, usa o modelo gerado por `python -m triagem.onnx_export`, desde
        que ele pertenca ao treino registrado nos metadados (mesma data de treino e mesmo
        SHA-256). O pipeline do scikit-learn e carregado com joblib (pickle): so e seguro
        com artefato gerado por este projeto. Nunca aponte para um arquivo desconhecido.

        Raises:
            ModeloIndisponivelError: artefato ausente, metadados invalidos, versao do
                scikit-learn diferente da usada no treino ou ONNX ausente/defasado.
        """
        usa_onnx = settings.backend is Backend.ONNX
        necessarios = [
            settings.metadata_path,
            settings.onnx_path if usa_onnx else settings.model_path,
        ]
        if not all(caminho.exists() for caminho in necessarios):
            raise ModeloIndisponivelError(
                f"Artefato nao encontrado em {settings.artifacts_dir}. "
                "Rode `python -m triagem.treino` primeiro"
                + (" e depois `python -m triagem.onnx_export`." if usa_onnx else ".")
            )
        metadata = _ler_metadata(settings)
        # A versao do scikit-learn importa nos dois motores: a normalizacao do texto usa a
        # lista de stop words dele, e o pickle so e valido na versao que o gerou.
        versao_treino = metadata["versoes"]["scikit_learn"]
        if versao_treino != sklearn.__version__:
            raise ModeloIndisponivelError(
                f"Modelo treinado com scikit-learn {versao_treino}, "
                f"mas o ambiente tem {sklearn.__version__}. Retreine ou alinhe as versoes."
            )
        versao = f"{metadata['tipo_modelo']}@{metadata['treinado_em']}"
        if usa_onnx:
            modelo: ModeloDeProbabilidades = _abrir_onnx(settings, metadata)
            versao += SUFIXO_VERSAO_ONNX
        else:
            modelo = _ModeloSklearn(joblib.load(settings.model_path))
        logger.info("Modelo carregado: %s", versao)
        return cls(modelo, float(metadata["limiar_urgente"]), versao)

    def classificar(self, texto: str) -> Resultado:
        """Classifica um laudo.

        Raises:
            TextoSemConteudoError: se nao sobrar nenhum termo util depois da normalizacao.
        """
        normalizado = normalizar(texto)
        if not normalizado:
            raise TextoSemConteudoError("O texto nao contem termos uteis para classificacao.")
        probabilidades = self._modelo.prever(normalizado)
        por_categoria = {
            c: float(p) for c, p in zip(self._modelo.classes, probabilidades, strict=True)
        }
        agregado = agregar_probabilidades(por_categoria)
        categoria = max(por_categoria, key=lambda c: por_categoria[c])
        return Resultado(
            urgencia=decidir_urgencia(agregado, self.limiar_urgente),
            probabilidades_urgencia=agregado,
            categoria_id=categoria,
            categoria_nome=CATEGORIAS[categoria],
            categoria_probabilidade=por_categoria[categoria],
        )


def _abrir_onnx(settings: Settings, metadata: dict[str, Any]) -> ModeloOnnx:
    """Abre o ONNX depois de conferir que ele e o do treino registrado nos metadados."""
    refazer = "Rode `python -m triagem.onnx_export`."
    registro = metadata.get("onnx")
    if not registro:
        raise ModeloIndisponivelError(f"Os metadados nao registram um modelo ONNX. {refazer}")
    if registro.get("treinado_em") != metadata["treinado_em"]:
        raise ModeloIndisponivelError(f"O modelo ONNX e de outro treino (defasado). {refazer}")
    if _sha256(settings.onnx_path) != registro.get("sha256"):
        raise ModeloIndisponivelError(f"O arquivo ONNX difere do registrado. {refazer}")
    try:
        return ModeloOnnx.abrir(settings.onnx_path, registro["classes"])
    except Exception as erro:  # o ONNX Runtime levanta tipos proprios (ex.: InvalidProtobuf)
        raise ModeloIndisponivelError(f"Modelo ONNX ilegivel: {erro}") from erro


def _sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def _ler_metadata(settings: Settings) -> dict[str, Any]:
    try:
        return json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as erro:
        raise ModeloIndisponivelError(f"Metadados ilegiveis: {erro}") from erro
