# Triagem de urgencia em laudos medicos

Tech Challenge - Fase 3 (Cloud and MLOps), PosTech FIAP Machine Learning Engineering.

Classificador de texto servido por API REST em container, com pipeline CI/CD,
orquestracao de treino e monitoramento.

> Em construcao. A decisao de arquitetura em nuvem, a API e as instrucoes completas
> de execucao entram nos proximos blocos. Progresso em `docs/roadmap.md`; decisoes e
> justificativas em `docs/decisoes_tecnicas.md`.

## Problema

Um hospital de referencia precisa triar automaticamente laudos medicos em texto,
classificando a urgencia em tres niveis: `normal`, `atencao` e `urgente`. Triagem rapida
reduz o tempo ate o atendimento dos casos criticos.

## Dataset

[Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
(11.550 resumos de treino e 2.888 de teste, 5 categorias de doenca), licenca
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Os dados nao ficam no
repositorio: sao baixados por `python -m triagem.dados` e validados por SHA-256.

### Mapeamento categoria -> urgencia

O dataset classifica por **categoria de doenca**, nao por urgencia. Para atender ao
enunciado foi definido um mapeamento explicito:

| Urgencia | Categorias do dataset | Justificativa |
|---|---|---|
| `urgente` | cardiovascular, sistema nervoso | condicoes com eventos agudos (ex.: infarto, AVC) |
| `atencao` | neoplasias, digestivo | exigem investigacao e acompanhamento |
| `normal` | condicoes patologicas gerais | categoria generica, sem foco em evento agudo |

> **Limitacao:** o mapeamento e uma heuristica de dominio, nao um criterio clinico.
> Neoplasias, por exemplo, tambem podem ser graves. A API devolve tambem a categoria
> original prevista, e o mapeamento fica isolado em `src/triagem/urgencia.py` para
> poder ser revisado sem retreinar o modelo.

## Ambiente de desenvolvimento

O projeto precisa ficar fora do OneDrive (ex.: `C:\dev\tech-challenge-fase3`).
Caminhos com acento quebram a criacao do `.venv` no Windows.

```powershell
poetry install
poetry run pre-commit install
cp .env.example .env
poetry run ruff check .
poetry run ruff format --check .
poetry run pytest
poetry run python -m triagem.dados
```
