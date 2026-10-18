# Decisoes tecnicas

## D-001 - Projeto fora do OneDrive
Local: `C:\dev\tech-challenge-fase3`. Caminhos com acento (`Area de Trabalho`) quebram
a inicializacao do `.venv` no Windows por problema de encoding (visto na fase 1).

## D-002 - Poetry, ruff e pytest
Poetry com `poetry.lock` para dependencias deterministicas. Ruff faz lint e formatacao
(regras E, F, W, I, B, UP, N, SIM, S). Pytest para testes. Layout `src/triagem`.

## D-003 - pre-commit com hooks locais para o ruff
Os hooks de ruff chamam `poetry run ruff`, entao a versao vem do `poetry.lock` e nao
fica duplicada no `.pre-commit-config.yaml`. Gitleaks ficou de fora dos hooks locais:
o hook exige toolchain Go e pode falhar no Windows. Reavaliado no Bloco 2: a varredura de
segredos roda no GitHub Actions (Linux), onde nao ha esse problema (ver D-030).

## D-004 - Dependencias entram por bloco
Bloco 0 so tem ferramentas de desenvolvimento. scikit-learn entra no 1b; FastAPI e uvicorn
no 1c; prometheus-client no 3; ONNX no 4. Airflow roda em container proprio e nao entra no
`pyproject.toml`, para evitar conflito de versoes com a API.

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
Candidatos: regressao logistica, Naive Bayes (simples e calibrado) e Random Forest
(TF-IDF). Selecao por validacao cruzada de 3 dobras no treino, criterio F1 macro de
urgencia. O teste e avaliado uma unica vez, depois da escolha. Resultado: Naive Bayes
calibrado (F1 0,637 na validacao cruzada; 0,632 no teste). A recomendacao inicial,
regressao logistica, perdeu para ele; foi a comparacao por dados que mostrou isso.
Tudo reproduzivel com `python -m triagem.treino`.

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
(`StringNormalizer`), que falhou ao carregar num ambiente sem o locale `en_US.UTF-8`
(situacao provavel em imagens slim; a confirmar no Docker, bloco 1d). Com a normalizacao
fora do pipeline, esse operador deixa de ser necessario. A concordancia entre ONNX e
scikit-learn sera medida no benchmark reproduzivel da Etapa 4. A mesma funcao e usada no
treino e na API.

## D-017 - Calibracao de probabilidades
O `ComplementNB` sozinho gera probabilidades planas, o que distorce a soma por nivel e o
limiar. A calibracao isotonica (`CalibratedClassifierCV`, 3 dobras) corrige: F1 de
urgencia de 0,543 (sem calibracao) para 0,637 (calibrado). Os dois candidatos ficam na
comparacao para que o efeito seja verificavel.

## D-018 - Limiar de decisao para `urgente`
Errar um caso urgente e o erro mais caro. O limiar de `urgente` e calibrado na validacao
cruzada para um recall alvo de 80% (padrao, configuravel por `--recall-alvo`) e usado na
API. No teste: recall 0,823 (contra 0,781 sem limiar) com precisao praticamente igual
(0,663 contra 0,660). Pontos de operacao: recall alvo 0,85 gera limiar 0,340 com recall
0,878 e precisao 0,637; 0,90 gera limiar 0,251 com recall 0,930 e precisao 0,599. Subir o
alvo e uma decisao clinica (quanto alarme falso o hospital aceita), nao tecnica.

## D-019 - Inferencia em thread unica
O Random Forest treinado com `n_jobs=-1` cria um pool de threads a cada chamada, o que
penaliza requisicoes individuais. O modelo e salvo com `n_jobs=1`. O benchmark da Etapa 4
usa `n_jobs=1` no modelo original para a comparacao com ONNX ser justa.

## D-020 - Reprodutibilidade numerica
Sementes fixas, dados com SHA-256 e `poetry.lock` garantem a mesma logica em qualquer
maquina, mas nao resultados identicos bit a bit: a soma em ponto flutuante das bibliotecas
de algebra linear depende do numero de threads e da CPU. Medido: mesma versao de
scikit-learn, numpy e scipy, o recall da regressao logistica foi 0,7146 com 1 thread e
0,7164 com 2 ou 8; o treino em Windows (Python 3.11) chegou a F1 de teste 0,628 contra
0,632 em Linux (Python 3.13). Python, numpy e scipy foram descartados como causa. Os testes
automatizados verificam comportamento, nao numeros exatos. Fixar as threads no codigo nao
garantiria numeros identicos entre Windows e Linux (bibliotecas BLAS diferentes) e
invalidaria os resultados ja registrados, entao ficou fora do 1d. O treino passou a rodar em
container (DAG do Airflow, D-029) e os numeros de referencia sao os dele: Linux, Python
3.11.16, F1 de teste 0,628, recall 0,823, limiar 0,415. Esse resultado coincide com o do
Windows (Python 3.11, mesma maquina) e difere do Linux com Python 3.13 (0,632), o que indica
que a diferenca acompanha a versao do Python ou a CPU e nao o sistema operacional; as duas
causas estao misturadas nos dados disponiveis e nao foram separadas. Os valores de D-012 e
D-018 (F1 0,632; recall 0,823 contra 0,781; pontos de operacao 0,85 e 0,90) vieram da execucao
Linux com Python 3.13 e nao foram remedidos no container; a tabela do README usa os do
container.

## D-021 - Contrato da API
`POST /predict` recebe `{"texto": ...}` (1 a 10.000 caracteres; o maior laudo do dataset
tem 3.999) e devolve urgencia, probabilidades por nivel, categoria clinica, limiar e versao
do modelo. `GET /health` confirma que o servico esta no ar. Entrada invalida e texto sem
termos uteis retornam 422 (sem termos uteis o modelo "preveria" so pelo prior, resultado
enganoso). Nomes em portugues, como o resto do projeto. Sem endpoint de lote: o desafio
nao pede.

## D-022 - Privacidade nos logs
Laudo e dado sensivel (LGPD). O conteudo do texto nunca e registrado: so metodo, rota,
status, duracao e `X-Request-ID`. Ha teste automatizado que falha se o texto aparecer no log.

## D-023 - API falha rapido e valida versao do modelo
O modelo e carregado uma vez, na inicializacao. Sem artefato, ou se a versao do
scikit-learn do artefato diferir da instalada, a API recusa subir em vez de responder
possivelmente errado. Por isso nao existe 503 no `/health`: se o servico responde, o
modelo esta carregado. O artefato e carregado com joblib (pickle), seguro apenas para
arquivos gerados por este projeto.

## D-024 - Servico de classificacao separado da API
`classificador.py` nao depende do FastAPI: a logica e testavel sem servidor e o modelo pode
ser trocado por ONNX (Etapa 4) sem mexer na API. Endpoints sincronos (`def`): a inferencia
usa CPU e, em `async def`, bloquearia o loop de eventos.

## D-025 - httpx2 em vez de httpx nos testes
O `TestClient` do Starlette atual marca o `httpx` como obsoleto e recomenda `httpx2`.
Dependencia apenas de desenvolvimento.

## D-026 - Modelo montado como volume, nao embutido na imagem
A imagem contem so codigo e dependencias; `artifacts/` e montado em `/app/artifacts`
(somente leitura). O modelo e gerado pelo treino (depois, pela DAG do Airflow, que grava no
mesmo volume) e a API apenas le. Retreinar nao exige rebuild, a imagem nao carrega dado
derivado de dataset e a mesma imagem serve a qualquer versao de modelo. Custo: sem o volume
o container recusa subir (D-023), que e o comportamento desejado. Treinar durante o
`docker build` foi descartado: baixaria o dataset a cada build e levaria minutos.

## D-027 - Dockerfile multi-stage, usuario nao-root e HEALTHCHECK
- Estagio de build instala so as dependencias de runtime (`poetry install --only main
  --no-root`) a partir do `poetry.lock`; o Poetry nao chega na imagem final.
- `pyproject.toml` e `poetry.lock` sao copiados antes do codigo para aproveitar o cache de
  camadas (licao da fase 2). O pacote e usado via `PYTHONPATH=/app/src`, sem instalar o
  projeto no ambiente virtual, porque o ambiente e copiado entre estagios.
- Base `python:3.11-slim`, a mesma minor usada no desenvolvimento; o `poetry.lock` foi
  resolvido nela.
- Usuario `app` (uid 10001) sem shell de login; codigo e dependencias pertencem ao root e
  ficam somente leitura para a API.
- `HEALTHCHECK` com o proprio Python (a imagem slim nao tem curl) batendo em `/health`.
- `.dockerignore` deixa fora `.git`, `.env*`, dados, artefatos, testes e docs: nenhum
  segredo ou dado entra no contexto de build.
- Tamanho medido da imagem: 490 MB (`docker images`, descomprimido). Nao foi otimizado
  alem do multi-stage e da base slim; a composicao por camada nao foi investigada.

## D-028 - Benchmark de latencia em dois niveis
`python -m triagem.benchmark` mede a classificacao em dois niveis: **inferencia** (chama o
`Classificador` direto: so o modelo) e **http** (chama `POST /predict` de uma API em
execucao, por exemplo no container: o que o cliente sente, com FastAPI e rede). Escolhas:
- Requisicoes sequenciais: o objetivo e a latencia de uma requisicao isolada, nao throughput.
- 100 chamadas de aquecimento descartadas (a primeira chamada e muito mais lenta que as
  seguintes) e 1.000 medicoes.
- Textos: 200 laudos reais do conjunto de teste, sorteados com semente fixa; laudo sintetico
  nao representaria o custo real, que depende do tamanho do texto.
- Cliente HTTP sobre socket (biblioteca padrao, sem dependencia nova) com conexao persistente.
- A API e consultada antes de medir: se estiver fora do ar, o script falha antes de gastar
  tempo. Se a versao do modelo da API diferir da local, ha aviso.
- Percentis por interpolacao linear (padrao do numpy). P50 e P95 sao estaveis entre
  execucoes; P99 e maximo oscilam mais, por serem caudas.
- O resultado depende da maquina. Serve para comparar o modelo original com o otimizado
  (Etapa 4) **na mesma maquina**, nao como valor absoluto de producao. O JSON gerado
  (`docs/benchmark_sklearn.json`) registra versoes, CPU e parametros.

Duas armadilhas encontradas na validacao em Windows com Docker Desktop, ambas corrigidas:
1. **Conexao ociosa.** A consulta inicial ao `/health` abria uma conexao que ficava parada
   enquanto a inferencia local era medida (segundos); o servidor (keep-alive de 5 s no
   uvicorn) a encerrava e a primeira chamada HTTP falhava. A conexao do `/health` agora e
   descartada e a medicao abre uma nova. Ha teste de regressao com servidor que encerra a
   conexao ociosa.
2. **Requisicao em dois pacotes.** O `http.client` envia cabecalhos e corpo em duas escritas.
   Medido contra a mesma API: de dentro do container, 13,2 ms (duas escritas) e 12,1 ms
   (uma escrita); do Windows, passando pelo Docker Desktop, **57,3 ms** (duas escritas) contra
   **11,3 ms** (uma escrita). O log do servidor mostrava 52 a 65 ms por requisicao, ou seja, o
   atraso ocorre antes de a classificacao comecar. A hipotese e a interacao do algoritmo de
   Nagle com ACK atrasado no caminho de rede do Docker Desktop; ela explica a ordem de
   grandeza, mas **nao foi comprovada**. O cliente do benchmark envia cada requisicao em uma
   unica escrita (como o `curl` faz com corpos pequenos), com `TCP_NODELAY`, para que o numero
   reflita a API e nao essa peculiaridade, que esconderia qualquer ganho da otimizacao.

Hipoteses descartadas por medicao: o numero de threads do BLAS/OpenMP no container nao muda
a latencia da classificacao (7,6 ms com threads padrao e 8,0 ms com 1 thread, em 8 CPUs), e a
classificacao dentro do container (~8 ms) equivale a do Windows (~9 ms).

## D-029 - Retreino orquestrado: Airflow standalone, staging e portao de qualidade
- **Logica fora da DAG.** As etapas ficam em `triagem.retreino` (testaveis sem Airflow, 9
  testes); a DAG so orquestra. O Airflow nao entra no `pyproject.toml` (D-004).
- **Staging, validacao e publicacao atomica.** O treino grava em
  `artifacts/.staging/<momento>` (mesmo sistema de arquivos do destino). So depois de validar o
  modelo ele e publicado com `os.replace`, arquivo por arquivo. O treino original gravava
  direto no arquivo final: uma falha no meio deixaria o modelo em uso corrompido. A publicacao
  recusa diretorios fora de `.staging` (nao move arquivos de outro lugar).
- **Portao de qualidade.** O modelo precisa carregar (inclui a conferencia da versao do
  scikit-learn, D-023), classificar um laudo de fumaca com probabilidades somando 1 e atingir
  no teste F1 de urgencia >= 0,55 e recall urgente >= 0,75. O modelo atual tem 0,63 e 0,82;
  os minimos deixam folga para a variacao numerica (D-020) e barram um treino quebrado.
  Configuravel no disparo da DAG. Os valores sao uma escolha minha de margem, nao um criterio
  clinico. Verificado: com `f1_minimo = 0.99`, `validar_modelo` falhou, `publicar_modelo` nao
  rodou e o modelo em uso permaneceu o mesmo.
- **Disparo manual** (`schedule=None`) e `max_active_runs=1` (nunca dois treinos juntos).
  Apenas `ler_dados`, a unica etapa que usa a rede, tem repeticao automatica.
- **Airflow `standalone`** (um container, SQLite) em `docker-compose.airflow.yml` separado do
  compose da API. Atende ao desafio e nao e arquitetura de producao; o equivalente em nuvem
  seria um job batch (D-014). A porta fica presa a `127.0.0.1`. O Airflow avisa que o
  gerenciador de autenticacao simples nao e recomendado para producao e imprime a senha
  gerada no log; para uso local foi mantida a autenticacao em vez de desativa-la.
- **Imagem** `apache/airflow:3.3.2-python3.11` com as dependencias de treino fixadas nas mesmas
  versoes do `poetry.lock` (`airflow/requirements-treino.txt`). Um teste compara as duas
  listas: se divergirem, a suite falha, em vez de a API recusar o modelo so em execucao. Codigo
  e DAGs entram por volume, para retreinar sem rebuild. `DockerOperator` foi descartado porque
  exigiria o socket do Docker.
- **Medido** (Docker Desktop, 8 CPUs): execucao completa em 4 min 01 s (`treinar`: 2 min 54 s).
  A API subiu com o modelo publicado pela DAG e classificou. A DAG tambem foi executada fora do
  container (Airflow 3.3.2, Python 3.13) durante o desenvolvimento, com o mesmo resultado.
- **Limite conhecido:** sao dois arquivos (modelo e metadados), trocados um apos o outro; ha
  uma janela de milissegundos entre as duas trocas. A API le o modelo so ao iniciar (D-023),
  entao so um restart nesse instante pegaria modelo novo com metadados antigos.
- **Nao tratado:** staging de modelos reprovados nao e apagado automaticamente (fica para
  diagnostico) e nao ha rollback para um modelo anterior.

## D-030 - CI no GitHub Actions
- **`ci.yml`** com tres jobs independentes: `lint` (ruff check e format), `test` (pytest) e
  `build` (docker build, sem push). **`seguranca.yml`** com Gitleaks sobre o historico completo
  (`fetch-depth: 0`), em push, pull request e semanalmente.
- **Python 3.11 apenas**, a versao da imagem; os testes verificam comportamento e nao numeros
  exatos (D-020), entao uma matriz de versoes custaria mais sem ganho para o projeto.
- **Poetry 2.4.1**, a mesma versao do Dockerfile. Permissoes minimas (`contents: read`),
  cancelamento de execucoes antigas e cache do ambiente virtual.
- **Limite do job `build`:** so prova que a imagem constroi; nao sobe a API, pois sem o modelo
  ela recusa iniciar (D-026). Um teste de fumaca do container exigiria treinar um modelo no CI
  (minutos, download do dataset) e foi deixado de fora; o comportamento da API e coberto pelo
  `pytest`.
- Dependabot descartado: o enunciado nao pede e abriria pull requests durante um prazo curto.
- **Validado:** o primeiro push com os workflows passou nos quatro jobs (CI em 1 min 44 s,
  varredura de segredos em 13 s). A varredura sem achados indica que as regras padrao do
  Gitleaks nao encontraram segredos no historico; nao e garantia de ausencia.

## D-031 - Metricas Prometheus na API
- **`prometheus-client`** (entrou no bloco 3, como previsto em D-004). Metricas: contador
  `http_requests_total{method,route,status}`, histograma `http_request_duration_seconds{route}`,
  histograma `triagem_inferencia_duration_seconds` (so a classificacao), contador
  `triagem_predicoes_total{urgencia}`, histograma `triagem_probabilidade_urgente` e gauge
  `triagem_modelo_info{versao}`.
- **Rota pelo template, nao pelo caminho cru.** A rota vem do roteamento do FastAPI
  (`/predict`); caminhos desconhecidos viram `desconhecida`. Usar o caminho cru criaria uma
  serie por URL e estouraria a memoria do Prometheus (explosao de cardinalidade). Os demais
  rotulos tem poucos valores (metodo, status, tres niveis de urgencia). Teste automatizado.
- **Privacidade (D-022):** nenhuma metrica recebe o texto do laudo. Teste automatizado.
- **`/metrics` fora da vazao e da documentacao.** A leitura feita pelo proprio Prometheus nao
  entra em `http_requests_total` (inflaria a vazao) e a rota nao aparece no Swagger.
- **Erro interno conta como 500.** O middleware registra a requisicao mesmo se o handler
  levantar excecao e a propaga. Teste com um classificador que falha de proposito.
- **Registro por aplicacao** (`CollectorRegistry` proprio) em vez do global: varias aplicacoes
  no mesmo processo (testes) nao colidem. `disable_created_metrics()` remove as series
  `*_created`, que o dashboard nao usa e dobrariam o numero de series.
- **Buckets** ajustados a faixa em que a API opera (5 ms a 2,5 s no HTTP; 1 ms a 1 s na
  inferencia, que fica perto de 10 ms), para os percentis terem resolucao onde importa.
- **Limites:** percentis do Prometheus sao estimativas por interpolacao entre buckets; assume
  uma unica worker do uvicorn; o sinal de drift (probabilidade de `urgente`) e indireto, pois
  nao ha rotulos em producao.

## D-032 - Stack de monitoramento: compose, Prometheus, Grafana e carga
- **Docker Compose** (`docker-compose.yml`) com `api`, `prometheus` e `grafana`. O Prometheus
  so inicia depois que a API esta saudavel (`depends_on` com `service_healthy`, usando o
  `HEALTHCHECK` da imagem). O modelo continua montado como volume somente leitura (D-026).
- **Portas so em `127.0.0.1`** e configuraveis pelo `.env` (`API_PORT`, `PROMETHEUS_PORT`,
  `GRAFANA_PORT`): na maquina de desenvolvimento 8000 e 3000 ja estavam ocupadas por outros
  containers, e o compose nao deve expor nada na rede local. Um teste confere que toda porta
  publicada comeca por `127.0.0.1`.
- **Senha do Grafana sem valor padrao.** `GRAFANA_ADMIN_PASSWORD` e obrigatoria
  (`${VAR:?...}`): sem ela o compose recusa subir, em vez de usar `admin/admin`. O custo e um
  passo a mais para quem clona o repositorio, documentado no README. Cadastro de usuarios e
  telemetria do Grafana desativados.
- **Provisionamento versionado.** Fonte de dados (uid fixo `prometheus`) e dashboard sao
  arquivos em `monitoring/grafana/`, montados somente leitura: o ambiente sobe pronto e
  reproduzivel, sem configuracao manual. Os 7 paineis atendem ao minimo de 3 do desafio e
  cobrem trafego, erros, latencia (HTTP e inferencia), mistura de urgencias, drift indireto e
  versao do modelo.
- **Testes de consistencia** (`tests/test_dashboard.py`): toda metrica citada numa consulta do
  dashboard existe no `/metrics` da API; as consultas usam o uid da fonte de dados
  provisionada; o Prometheus coleta `api:8000`. Renomear uma metrica sem atualizar o JSON
  deixaria um painel vazio sem erro algum; agora a suite falha.
- **Gerador de carga** (`python -m triagem.carga`): taxa constante e sequencial, laudos reais
  do conjunto de teste e 10% de entradas invalidas (texto vazio ou sem conteudo, ambos 422).
  Serve para gerar trafego visivel, nao para medir capacidade (isso e do benchmark, D-028).
- **Versoes:** Prometheus v3.13.4 e Grafana 12.4.13 (tags fixas, sem `latest`), ambas
  baixadas e executadas com sucesso no Docker Desktop. Dados do Prometheus guardados por 7 dias
  em volume nomeado.
- **Nao tratado:** alertas, autenticacao no Prometheus, coleta de metricas do host/container e
  varias instancias da API.
