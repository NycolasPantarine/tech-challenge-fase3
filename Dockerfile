# syntax=docker/dockerfile:1

# Imagem de inferencia da API de triagem. O modelo NAO fica na imagem: e montado como
# volume em /app/artifacts (ver README, secao Docker e docs/decisoes_tecnicas.md D-026).

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------
# Estagio 1 - build: instala as dependencias de runtime a partir do poetry.lock.
# O Poetry existe so aqui e nao chega na imagem final.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ARG POETRY_VERSION=2.4.1

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Copiar so os arquivos de dependencias primeiro: enquanto eles nao mudarem, o Docker
# reaproveita esta camada e o rebuild apos editar o codigo leva segundos.
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

# ---------------------------------------------------------------------------
# Estagio 2 - runtime: so o ambiente virtual pronto e o codigo da aplicacao.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# Usuario sem privilegios e sem shell de login. Os arquivos da aplicacao continuam
# pertencendo ao root (somente leitura para o usuario da API).
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRIAGEM_ARTIFACTS_DIR=/app/artifacts

USER app

EXPOSE 8000

# Sem curl na imagem slim: a checagem usa o proprio Python. urlopen levanta excecao em
# qualquer status que nao seja 2xx, o que marca o container como unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "triagem.api:criar_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
