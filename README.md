# Triagem de urgencia em laudos medicos

Tech Challenge - Fase 3 (Cloud and MLOps), PosTech FIAP Machine Learning Engineering.

Classificador de texto servido por API REST em container, com pipeline CI/CD,
orquestracao de treino e monitoramento.

> Em construcao. O monitoramento e a decisao de arquitetura em nuvem entram nos proximos
> blocos. Progresso em `docs/roadmap.md`; decisoes e justificativas em
> `docs/decisoes_tecnicas.md`.

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
e derivada somando as probabilidades de cada grupo. Quatro candidatos sao comparados por
validacao cruzada (3 dobras) **somente no treino**; o conjunto de teste e usado uma unica
vez, apos a escolha. O criterio de selecao e o F1 macro de urgencia.

| Candidato | F1 urgencia | Recall urgente | Acuracia (5 categorias) |
|---|---|---|---|
| Regressao logistica | 0,599 | 0,716 | 0,556 |
| Naive Bayes (sem calibracao) | 0,543 | 0,856 | 0,600 |
| **Naive Bayes calibrado** | **0,637** | 0,762 | **0,602** |
| Random Forest | 0,452 | 0,826 | 0,517 |

O Naive Bayes sem calibracao tem boa acuracia de categoria, mas probabilidades pouco
confiaveis (planas), que distorcem a soma por nivel de urgencia e o limiar. A calibracao
isotonica corrige isso e leva o F1 de urgencia de 0,543 para 0,637. Os dois candidatos
estao na comparacao para que esse efeito possa ser verificado com `python -m triagem.treino`.

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

### Reprodutibilidade

O treino e deterministico no codigo (sementes fixas, dados validados por SHA-256, versoes
travadas no `poetry.lock`), mas os numeros **nao sao identicos bit a bit entre maquinas**:
a soma em ponto flutuante das bibliotecas de algebra linear muda com o numero de threads e
com a CPU. Os valores acima foram obtidos em Linux com Python 3.13; em Windows com
Python 3.11 o mesmo codigo chegou a F1 de teste 0,628 (limiar 0,415). A escolha do modelo
e a conclusao sao as mesmas; espere diferencas de ate cerca de 0,005. As versoes usadas em
cada treino ficam registradas em `artifacts/metadata.json`.

## API

Servidor FastAPI. Documentacao interativa (Swagger) em `http://localhost:8000/docs`.

```powershell
poetry run uvicorn triagem.api:criar_app --factory --port 8000
```

A API carrega o modelo de `artifacts/` ao iniciar e **recusa subir** se o artefato nao
existir ou se a versao do scikit-learn for diferente da usada no treino.

**`POST /predict`**: classifica um laudo (1 a 10.000 caracteres).

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body '{"texto": "Patient with acute myocardial infarction and chest pain"}'
```

Resposta:

```json
{
  "urgencia": "urgente",
  "probabilidades_urgencia": {"normal": 0.12, "atencao": 0.21, "urgente": 0.67},
  "categoria": {"id": 4, "nome": "cardiovascular diseases", "probabilidade": 0.58},
  "limiar_urgente": 0.415,
  "versao_modelo": "naive_bayes_calibrado@2026-10-05T20:49:00+00:00"
}
```

Os valores acima ilustram o formato. Entrada invalida (vazia, acima do limite, sem
termos uteis) retorna `422`.

**`GET /health`**: confirma que o servico esta no ar e informa a versao do modelo.

> **Privacidade:** laudos sao dados sensiveis. O conteudo do texto **nunca** e registrado
> em log; so metodo, rota, status, duracao e um identificador da requisicao
> (`X-Request-ID`). Ferramenta de apoio a decisao: nao substitui avaliacao clinica.

## Docker

A imagem contem so o codigo e as dependencias de runtime. O **modelo nao fica na imagem**:
`artifacts/` e montado como volume somente leitura (decisao D-026), o que permite retreinar
sem rebuild. Sem o volume o container recusa subir.

```powershell
poetry run python -m triagem.treino   # gera artifacts/ (uma vez)
docker build -t triagem-api:dev .
docker run --rm -d --name triagem-api -p 8000:8000 -v "${PWD}\artifacts:/app/artifacts:ro" triagem-api:dev
```

Se a porta 8000 estiver ocupada, troque o numero da esquerda (`-p 8010:8000`). Conferir:

```powershell
docker ps                                       # STATUS deve chegar a "healthy"
Invoke-RestMethod http://localhost:8000/health
docker stop triagem-api
```

Caracteristicas da imagem (D-027): build em dois estagios (o Poetry nao chega na imagem
final), base `python:3.11-slim`, execucao com usuario sem privilegios (uid 10001),
`HEALTHCHECK` em `/health` e nenhum segredo ou dado no contexto de build (`.dockerignore`).

Tamanho medido da imagem: 490 MB (`docker images`, descomprimido).

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
