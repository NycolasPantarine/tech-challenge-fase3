# Roadmap - Tech Challenge Fase 3

Prazo oficial: 27/10/2026, podendo ser antecipado conforme a disponibilidade.
Regra de trabalho: so avanca de bloco quando o atual estiver 100% validado.

| Bloco | Datas | Etapa | Status |
|---|---|---|---|
| 0 | 05/10 | Setup do projeto | concluido |
| 1 | 05-10/10 | Etapa 1: dataset, modelo base, API, Docker, baseline de latencia | concluido |
| 2 | 14/10 | Etapa 2: testes, GitHub Actions, DAG Airflow | concluido |
| 3 | 15-18/10 | Etapa 3: Prometheus, Grafana, docker-compose | concluido |
| 4 | 19-22/10 | Etapa 4: ONNX, benchmark original vs otimizado | pendente |
| 5 | 23-25/10 | README final, validacao do zero, video STAR | pendente |
| - | 26/10 | Reserva | - |
| - | 27/10 | Entrega | - |

## Sub-blocos do Bloco 1

| Sub-bloco | Entrega | Status |
|---|---|---|
| 1a | config, `.dockerignore`, download verificado do dataset, mapeamento de urgencia, testes | concluido |
| 1b | comparacao de modelos, treino, avaliacao e artefato gerado | concluido |
| 1c | API FastAPI (`/predict`, `/health`) com testes | concluido |
| 1d | Dockerfile multi-stage, usuario nao-root, HEALTHCHECK, modelo como volume | concluido |
| 1e | baseline de latencia (P50/P95/P99) e README | concluido |

## Sub-blocos do Bloco 2

| Sub-bloco | Entrega | Status |
|---|---|---|
| 2a | workflows do GitHub Actions: lint, testes, build da imagem e varredura de segredos | concluido |
| 2b | DAG do Airflow (ler dados, treinar, validar, publicar) com portao de qualidade | concluido |

## Sub-blocos do Bloco 3

| Sub-bloco | Entrega | Status |
|---|---|---|
| 3a | metricas Prometheus na API (`/metrics`) com testes | concluido |
| 3b | `docker-compose.yml` com API e Prometheus, gerador de carga | concluido |
| 3c | Grafana provisionado, dashboard de 7 paineis e print | concluido |
| 3d | documentacao: README, D-031 e D-032, roadmap | concluido |

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
dados, estrategia de nuvem, modelo e API estao registrados em `docs/decisoes_tecnicas.md`.
