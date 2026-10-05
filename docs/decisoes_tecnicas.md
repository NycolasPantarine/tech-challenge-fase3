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
Bloco 0 so tem ferramentas de desenvolvimento. FastAPI e afins entram no Bloco 1c;
scikit-learn no 1b; prometheus-client no 3; ONNX no 4. Airflow roda em container proprio
e nao entra no `pyproject.toml`, para evitar conflito de versoes com a API.

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
transparente. Isso exige probabilidades calibradas (ver D-017).

## D-012 - Modelo escolhido por dados
Candidatos: regressao logistica, Naive Bayes calibrado e Random Forest (TF-IDF).
Selecao por validacao cruzada de 3 dobras no treino, criterio F1 macro de urgencia. O teste
e avaliado uma unica vez, depois da escolha. Resultado: Naive Bayes calibrado (F1 0,637 na
validacao cruzada; 0,632 no teste). A recomendacao inicial, regressao logistica, perdeu
para ele; foi a comparacao por dados que mostrou isso.

## D-013 - Dados e modelo nao sao versionados
Dados baixados por script com SHA-256 fixo; modelo gerado pelo treino. No git entra so
codigo. Sem DVC: o desafio nao exige e o download verificado garante reprodutibilidade.

## D-014 - Estrategia de nuvem (a documentar no README)
GCP Cloud Run para a API em tempo real e job batch para retreino. Apenas documentada,
sem deploy.

## D-015 - Download seguro de dados
Somente https, timeout, verificacao de SHA-256, gravacao atomica (nunca deixa arquivo
parcial) e re-download se o arquivo local estiver corrompido.

## D-016 - Normalizacao de texto fora do pipeline
Minusculas, tokenizacao e remocao de stop words ficam em `triagem.texto.normalizar`. O
`TfidfVectorizer` com `lowercase=True` ou `stop_words` vira um operador ONNX
(`StringNormalizer`) que exige locale do sistema e falha em imagens `python:3.11-slim`
(erro reproduzido). Com a normalizacao fora, o ONNX converte e concordou com o
scikit-learn em 99,8% dos rotulos (diferenca maxima de probabilidade de 0,016) no teste
preliminar. A mesma funcao e usada no treino e na API.

## D-017 - Calibracao de probabilidades
O `ComplementNB` sozinho gera probabilidades planas (media da maior probabilidade: 0,43,
contra 0,67 da regressao logistica), o que distorce a soma por nivel e o limiar. A
calibracao isotonica (`CalibratedClassifierCV`, 3 dobras) corrigiu: F1 de urgencia foi de
0,543 para 0,637.

## D-018 - Limiar de decisao para `urgente`
Errar um caso urgente e o erro mais caro. O limiar de `urgente` e calibrado na validacao
cruzada para um recall alvo de 80% (padrao, configuravel) e usado na API. No teste: recall
0,823 (contra 0,781 sem limiar) com precisao praticamente igual (0,663 contra 0,660).

## D-019 - Inferencia em thread unica
O Random Forest treinado com `n_jobs=-1` cria um pool de threads a cada chamada, o que
penaliza requisicoes individuais. O modelo e salvo com `n_jobs=1`. O benchmark da Etapa 4
usa `n_jobs=1` no modelo original para a comparacao com ONNX ser justa.
