# Triagem de urgencia em laudos medicos

Tech Challenge - Fase 3 (Cloud and MLOps), PosTech FIAP Machine Learning Engineering.

Classificador de texto servido por API REST em container, com pipeline CI/CD,
orquestracao de treino e monitoramento.

> Em construcao. A decisao de arquitetura em nuvem e as instrucoes de execucao
> entram nos proximos blocos. Progresso em `docs/roadmap.md`.

## Ambiente de desenvolvimento

O projeto precisa ficar fora do OneDrive (ex.: `C:\dev\tech-challenge-fase3`).
Caminhos com acento quebram a criacao do `.venv` no Windows.

```powershell
poetry install
poetry run pre-commit install
poetry run ruff check .
poetry run ruff format --check .
poetry run pytest
```
