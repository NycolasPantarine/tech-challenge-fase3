# Triagem de urgencia em laudos medicos

Tech Challenge - Fase 3 (Cloud and MLOps), PosTech FIAP Machine Learning Engineering.

Classificador de texto servido por API REST em container, com pipeline CI/CD,
orquestracao de treino e monitoramento.

> Em construcao. A API, o Docker, o monitoramento e a decisao de arquitetura em nuvem
> entram nos proximos blocos. Progresso em `docs/roadmap.md`; decisoes e justificativas
> em `docs/decisoes_tecnicas.md`.

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

## Modelo

O modelo preve as 5 categorias do dataset (TF-IDF com unigramas e bigramas) e a urgencia
e derivada somando as probabilidades de cada grupo. Tres candidatos foram comparados por
validacao cruzada (3 dobras) **somente no treino**; o conjunto de teste foi usado uma unica
vez, apos a escolha. O criterio de selecao e o F1 macro de urgencia.

| Candidato | F1 urgencia | Recall urgente | Acuracia (5 categorias) |
|---|---|---|---|
| Regressao logistica | 0,599 | 0,716 | 0,556 |
| **Naive Bayes calibrado** | **0,637** | 0,762 | **0,602** |
| Random Forest | 0,452 | 0,826 | 0,517 |

O Naive Bayes sem calibracao parecia o pior candidato: suas probabilidades sao pouco
confiaveis e distorcem a soma por nivel de urgencia. A calibracao isotonica corrigiu isso.

### Resultado final (conjunto de teste)

Em triagem, deixar de sinalizar um caso urgente e o erro mais caro. Por isso o limiar de
`urgente` e calibrado na validacao cruzada para um **recall alvo de 80%** (limiar 0,414).

| F1 urgencia | Recall urgente | Precisao urgente | Acuracia urgencia | Acuracia (5 categorias) |
|---|---|---|---|---|
| 0,632 | 0,823 | 0,663 | 0,658 | 0,596 |

> **Limitacoes assumidas:** a classe `normal` (condicoes patologicas gerais) tem recall de
> apenas 31%: muitos laudos `normal` sao escalados para `urgente`. Em triagem esse e o lado
> seguro do erro, mas gera carga extra de revisao. Este dataset e dificil (5 categorias com
> fronteiras difusas); o foco do trabalho e a engenharia do ciclo de vida do modelo, nao
> maximizar a acuracia.

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
poetry run python -m triagem.dados    # baixa e valida o dataset
poetry run python -m triagem.treino   # compara modelos, treina e gera artifacts/
```
