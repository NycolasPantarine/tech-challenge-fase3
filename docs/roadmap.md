# Roadmap - Tech Challenge Fase 3

Prazo assumido: 27/10/2026 (fim do periodo da fase no material). A confirmar.
Regra de trabalho: so avanca de bloco quando o atual estiver 100% validado.

| Bloco | Datas | Etapa | Status |
|---|---|---|---|
| 0 | 05/10 | Setup do projeto | concluido |
| 1 | 06-09/10 | Etapa 1: dataset, modelo base, API, Docker, baseline de latencia | pendente |
| 2 | 10-14/10 | Etapa 2: testes, GitHub Actions, DAG Airflow | pendente |
| 3 | 15-18/10 | Etapa 3: Prometheus, Grafana, docker-compose | pendente |
| 4 | 19-22/10 | Etapa 4: ONNX, benchmark original vs otimizado | pendente |
| 5 | 23-25/10 | README final, validacao do zero, video STAR | pendente |
| - | 26/10 | Reserva | - |
| - | 27/10 | Entrega | - |

## Bloco 0 - criterios de aceite

- [x] Projeto em `C:\dev\tech-challenge-fase3` (fora do OneDrive)
- [x] `poetry install` conclui sem erro
- [x] `poetry run ruff check .` limpo
- [x] `poetry run ruff format --check .` limpo
- [x] `poetry run pytest` passa (1 teste de fumaca)
- [x] `poetry run pre-commit run --all-files` passa
- [x] Repositorio git inicializado com primeiro commit semantico
- [x] Repositorio publico no GitHub e push feito

## Decisoes em aberto

- Dataset e mapeamento para niveis de urgencia
- Modelo base (regressao logistica vs Random Forest)
- Estrategia de nuvem documentada no README
- Versionamento de dados e artefatos de modelo (git, DVC ou gerado no build)
