"""Servico de classificacao: carrega o artefato treinado e classifica laudos.

Nao depende do FastAPI, entao a logica pode ser testada sem subir servidor e o modelo pode
ser trocado (ex.: por ONNX, na Etapa 4) sem mexer na API.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from triagem.config import Settings
from triagem.texto import normalizar
from triagem.urgencia import CATEGORIAS, Urgencia, agregar_probabilidades, decidir_urgencia

logger = logging.getLogger(__name__)


class ModeloIndisponivelError(RuntimeError):
    """O artefato do modelo esta ausente, ilegivel ou incompativel com o ambiente."""


class TextoSemConteudoError(ValueError):
    """O texto nao tem nenhum termo util depois da normalizacao."""


@dataclass(frozen=True)
class Resultado:
    """Resultado de uma classificacao."""

    urgencia: Urgencia
    probabilidades_urgencia: dict[Urgencia, float]
    categoria_id: int
    categoria_nome: str
    categoria_probabilidade: float


class Classificador:
    """Pipeline treinado, limiar de urgencia e versao do modelo."""

    def __init__(self, pipeline: Pipeline, limiar_urgente: float, versao: str) -> None:
        self._pipeline = pipeline
        self._classes = [int(c) for c in pipeline.classes_]
        self.limiar_urgente = limiar_urgente
        self.versao = versao

    @classmethod
    def carregar(cls, settings: Settings) -> "Classificador":
        """Carrega o artefato gerado por `python -m triagem.treino`.

        O pipeline e carregado com joblib (pickle): so e seguro com artefato gerado por este
        projeto. Nunca aponte para um arquivo de origem desconhecida.

        Raises:
            ModeloIndisponivelError: artefato ausente, metadados invalidos ou versao do
                scikit-learn diferente da usada no treino.
        """
        if not settings.model_path.exists() or not settings.metadata_path.exists():
            raise ModeloIndisponivelError(
                f"Artefato nao encontrado em {settings.artifacts_dir}. "
                "Rode `python -m triagem.treino` primeiro."
            )
        metadata = _ler_metadata(settings)
        versao_treino = metadata["versoes"]["scikit_learn"]
        if versao_treino != sklearn.__version__:
            raise ModeloIndisponivelError(
                f"Modelo treinado com scikit-learn {versao_treino}, "
                f"mas o ambiente tem {sklearn.__version__}. Retreine ou alinhe as versoes."
            )
        pipeline = joblib.load(settings.model_path)
        versao = f"{metadata['tipo_modelo']}@{metadata['treinado_em']}"
        logger.info("Modelo carregado: %s", versao)
        return cls(pipeline, float(metadata["limiar_urgente"]), versao)

    def classificar(self, texto: str) -> Resultado:
        """Classifica um laudo.

        Raises:
            TextoSemConteudoError: se nao sobrar nenhum termo util depois da normalizacao.
        """
        normalizado = normalizar(texto)
        if not normalizado:
            raise TextoSemConteudoError("O texto nao contem termos uteis para classificacao.")
        probabilidades = self._pipeline.predict_proba([normalizado])[0]
        por_categoria = {c: float(p) for c, p in zip(self._classes, probabilidades, strict=True)}
        agregado = agregar_probabilidades(por_categoria)
        categoria = max(por_categoria, key=lambda c: por_categoria[c])
        return Resultado(
            urgencia=decidir_urgencia(agregado, self.limiar_urgente),
            probabilidades_urgencia=agregado,
            categoria_id=categoria,
            categoria_nome=CATEGORIAS[categoria],
            categoria_probabilidade=por_categoria[categoria],
        )


def _ler_metadata(settings: Settings) -> dict[str, Any]:
    try:
        return json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as erro:
        raise ModeloIndisponivelError(f"Metadados ilegiveis: {erro}") from erro
