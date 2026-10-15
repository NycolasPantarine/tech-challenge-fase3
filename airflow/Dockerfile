# Imagem do Airflow para o retreino. O codigo (src/) e as DAGs entram por volume, para
# retreinar com o codigo atual sem rebuild. Aqui entram so as dependencias de treino, nas
# mesmas versoes do poetry.lock (ver requirements-treino.txt).
FROM apache/airflow:3.3.2-python3.11

COPY requirements-treino.txt /tmp/requirements-treino.txt
RUN pip install --no-cache-dir -r /tmp/requirements-treino.txt

ENV PYTHONPATH=/opt/airflow/src
