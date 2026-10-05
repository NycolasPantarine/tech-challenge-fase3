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


## D-010 - Dataset: Medical Abstracts TC Corpus
Sugerido pelo enunciado e com mais de 2.000 amostras. Baixado do repositorio original
(GitHub), nao do Kaggle. Licenca CC BY-SA 3.0: atribuicao no README e dados nao
redistribuidos (ver D-013).

## D-011 - Urgencia derivada da categoria clinica
O dataset nao tem rotulo de urgencia. O modelo preve as 5 categorias e a urgencia
(`normal`, `atencao`, `urgente`) e derivada por mapeamento explicito
(`src/triagem/urgencia.py`), somando as probabilidades de cada grupo. Assim a API pode
devolver categoria e urgencia, o mapeamento e revisavel sem retreino e a limitacao fica
transparente. Treinar direto nos 3 niveis sera comparado no bloco 1b.

## D-012 - Modelo escolhido por dados
Candidatos: regressao logistica, LinearSVC e Random Forest (TF-IDF). A escolha usa
validacao cruzada com foco no recall da classe `urgente` (erro mais caro), e nao
preferencia previa. Testes preliminares: acuracia 48% a 53% em 5 classes; ONNX reduz
a latencia em ~25% (regressao logistica) e ~150x (Random Forest, a confirmar com
`n_jobs=1` no modelo original).

## D-013 - Dados e modelo nao sao versionados
Dados baixados por script com SHA-256 fixo; modelo gerado pelo treino. No git entra so
codigo. Sem DVC: o desafio nao exige e o download verificado garante reprodutibilidade.

## D-014 - Estrategia de nuvem (a documentar no README)
GCP Cloud Run para a API em tempo real e job batch para retreino. Apenas documentada,
sem deploy.

## D-015 - Download seguro de dados
Somente https, timeout, verificacao de SHA-256, gravacao atomica (nunca deixa arquivo
parcial) e re-download se o arquivo local estiver corrompido.
