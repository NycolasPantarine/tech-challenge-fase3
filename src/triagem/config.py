"""Configuracao do projeto, lida de variaveis de ambiente com padroes seguros."""

import os
from dataclasses import dataclass
from pathlib import Path

ENV_DATA_DIR = "TRIAGEM_DATA_DIR"
ENV_ARTIFACTS_DIR = "TRIAGEM_ARTIFACTS_DIR"


@dataclass(frozen=True)
class Settings:
    """Parametros de execucao. Imutavel para evitar alteracao acidental em runtime."""

    data_dir: Path
    artifacts_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        """Monta a configuracao a partir do ambiente, com padroes relativos ao diretorio atual."""
        return cls(
            data_dir=Path(os.environ.get(ENV_DATA_DIR, "data")),
            artifacts_dir=Path(os.environ.get(ENV_ARTIFACTS_DIR, "artifacts")),
        )

    @property
    def raw_dir(self) -> Path:
        """Diretorio dos dados brutos baixados."""
        return self.data_dir / "raw"
