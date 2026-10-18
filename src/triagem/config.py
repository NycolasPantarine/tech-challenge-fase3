"""Configuracao do projeto, lida de variaveis de ambiente com padroes seguros."""

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

ENV_DATA_DIR = "TRIAGEM_DATA_DIR"
ENV_ARTIFACTS_DIR = "TRIAGEM_ARTIFACTS_DIR"
ENV_BACKEND = "TRIAGEM_BACKEND"


class Backend(StrEnum):
    """Motor de inferencia: o pipeline original do scikit-learn ou o modelo ONNX."""

    SKLEARN = "sklearn"
    ONNX = "onnx"


def _ler_backend(valor: str) -> Backend:
    try:
        return Backend(valor)
    except ValueError:
        validos = ", ".join(b.value for b in Backend)
        raise ValueError(f"{ENV_BACKEND}={valor!r} invalido; use um de: {validos}.") from None


@dataclass(frozen=True)
class Settings:
    """Parametros de execucao. Imutavel para evitar alteracao acidental em runtime."""

    data_dir: Path
    artifacts_dir: Path
    backend: Backend = Backend.SKLEARN

    @classmethod
    def from_env(cls) -> "Settings":
        """Monta a configuracao a partir do ambiente, com padroes relativos ao diretorio atual.

        Raises:
            ValueError: `TRIAGEM_BACKEND` com valor desconhecido.
        """
        return cls(
            data_dir=Path(os.environ.get(ENV_DATA_DIR, "data")),
            artifacts_dir=Path(os.environ.get(ENV_ARTIFACTS_DIR, "artifacts")),
            backend=_ler_backend(os.environ.get(ENV_BACKEND, Backend.SKLEARN.value)),
        )

    @property
    def raw_dir(self) -> Path:
        """Diretorio dos dados brutos baixados."""
        return self.data_dir / "raw"

    @property
    def model_path(self) -> Path:
        """Arquivo do pipeline treinado (gerado pelo treino, nao versionado)."""
        return self.artifacts_dir / "modelo.joblib"

    @property
    def onnx_path(self) -> Path:
        """Modelo convertido para ONNX (gerado por `python -m triagem.onnx_export`)."""
        return self.artifacts_dir / "modelo.onnx"

    @property
    def metadata_path(self) -> Path:
        """Metadados do treino: versoes, hash dos dados, metricas e limiar."""
        return self.artifacts_dir / "metadata.json"
