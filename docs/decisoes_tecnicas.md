# Decisoes tecnicas

## D-001 - Projeto fora do OneDrive
Local: `C:\dev\tech-challenge-fase3`. Caminhos com acento (`Area de Trabalho`) quebram
a inicializacao do `.venv` no Windows por problema de encoding (visto na fase 1).

## D-002 - Poetry, ruff e pytest
Poetry com `poetry.lock` para dependencias deterministicas. Ruff faz lint e formatacao
(regras E, F, W, I, B, UP, N, SIM, S). Pytest para testes. Layout `src/triagem`.

## D-003 - pre-commit com hooks locais para o ruff
Os hooks de ruff chamam `poetry run ruff`, entao a versao vem do `poetry.lock` e nao
fica duplicada no `.pre-commit-config.yaml`. Gitleaks ficou de fora por enquanto:
o hook exige toolchain Go e pode falhar no Windows. Reavaliar no Bloco 2.

## D-004 - Dependencias entram por bloco
Bloco 0 so tem ferramentas de desenvolvimento. FastAPI, scikit-learn e afins entram
no Bloco 1; prometheus-client no 3; ONNX no 4. Airflow roda em container proprio e
nao entra no `pyproject.toml`, para evitar conflito de versoes com a API.

## D-005 - `.gitignore` com padroes ancorados
Padroes de pasta usam `/` na raiz. Na fase 2 um `models/` solto bloqueou `src/models/`.

## D-006 - `.gitattributes` com `eol=lf`
Evita que o Git do Windows converta shell scripts e Dockerfiles para CRLF.

## D-007 - Python 3.11
O projeto exige Python `>=3.11,<3.14` e o ruff usa `py311`. O container usara
`python:3.11-slim`, o mesmo da fase 2, para que ambiente local e container nao divirjam.

## D-008 - Praticas herdadas da fase 2
Reaproveitadas: modulos com responsabilidade unica (config, dados, modelo), type hints e
docstrings em todas as funcoes, configuracao por variaveis de ambiente (`.env.example`),
`.dockerignore` desde o primeiro build, cache de camadas no Dockerfile, validacao clonando
do zero.
Nao reaproveitadas: `git init` dentro da imagem, `.env` copiado para a imagem e Poetry
instalado na imagem final. A fase 3 usa Docker multi-stage, usuario nao-root e configuracao
via variaveis de ambiente.

## D-009 - Repositorio publico
Criado como publico para que o avaliador acesse o codigo, o historico de commits e a
documentacao sem precisar de convite.
