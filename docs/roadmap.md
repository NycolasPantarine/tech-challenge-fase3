# Roadmap - Tech Challenge Fase 3

Prazo oficial: 27/10/2026, podendo ser antecipado conforme a disponibilidade.
Regra de trabalho: so avanca de bloco quando o atual estiver 100% validado.

| Bloco | Datas | Etapa | Status |
|---|---|---|---|
| 0 | 05/10 | Setup do projeto | concluido |
| 1 | 06-09/10 | Etapa 1: dataset, modelo base, API, Docker, baseline de latencia | em andamento (1b) |
| 2 | 10-14/10 | Etapa 2: testes, GitHub Actions, DAG Airflow | pendente |
| 3 | 15-18/10 | Etapa 3: Prometheus, Grafana, docker-compose | pendente |
| 4 | 19-22/10 | Etapa 4: ONNX, benchmark original vs otimizado | pendente |
| 5 | 23-25/10 | README final, validacao do zero, video STAR | pendente |
| - | 26/10 | Reserva | - |
| - | 27/10 | Entrega | - |

## Sub-blocos do Bloco 1

| Sub-bloco | Entrega | Status |
|---|---|---|
| 1a | config, `.dockerignore`, download verificado do dataset, mapeamento de urgencia, testes | concluido |
| 1b | comparacao de modelos, treino, avaliacao e artefato gerado | concluido (ajuste de reprodutibilidade em validacao) |
| 1c | API FastAPI (`/predict`, `/health`) com testes | pendente |
| 1d | Dockerfile multi-stage, usuario nao-root, HEALTHCHECK, threads fixas no treino | pendente |
| 1e | baseline de latencia (P50/P95/P99) e README | pendente |

## Bloco 0 - criterios de aceite (concluido)

- [x] Projeto em `C:\dev\tech-challenge-fase3` (fora do OneDrive)
- [x] `poetry install` conclui sem erro
- [x] `poetry run ruff check .` limpo
- [x] `poetry run ruff format --check .` limpo
- [x] `poetry run pytest` passa (1 teste de fumaca)
- [x] `poetry run pre-commit run --all-files` passa
- [x] Repositorio git inicializado com primeiro commit semantico
- [x] Repositorio publico no GitHub e push feito

## Decisoes tomadas

Dataset, mapeamento de urgencia, criterio de escolha do modelo, versionamento de
dados, estrategia de nuvem e detalhes do modelo estao registrados em
`docs/decisoes_tecnicas.md`.
