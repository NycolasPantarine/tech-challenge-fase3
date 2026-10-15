"""DAG de retreino do classificador de urgencia.

ler_dados -> treinar -> validar_modelo -> publicar_modelo

- ler_dados: baixa (se preciso) e valida o dataset (SHA-256) e confere os CSVs.
- treinar: compara candidatos, escolhe o melhor e grava em um staging.
- validar_modelo: portao de qualidade (carrega, classifica e confere as metricas minimas).
- publicar_modelo: troca atomicamente o modelo que a API le. Se a validacao falhar, o modelo
  em uso permanece intacto.

A logica fica em `triagem.retreino`; esta DAG so orquestra. O disparo e manual: retreinar
sem motivo nao melhora o modelo, e o dataset nao muda.
"""

from datetime import datetime, timedelta

from airflow.sdk import dag, task

# Valores iguais aos de triagem.retreino; ficam literais aqui para a DAG ser interpretada
# sem importar scikit-learn. Podem ser alterados no disparo manual da DAG.
F1_URGENCIA_MINIMO = 0.55
RECALL_URGENTE_MINIMO = 0.75


@dag(
    dag_id="treino_triagem",
    description="Retreina o classificador de urgencia e publica o modelo se for aprovado.",
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,  # nunca dois treinos ao mesmo tempo
    params={"f1_minimo": F1_URGENCIA_MINIMO, "recall_minimo": RECALL_URGENTE_MINIMO},
    tags=["triagem", "mlops"],
)
def treino_triagem():
    @task(retries=2, retry_delay=timedelta(seconds=30))  # repete so a etapa que usa a rede
    def ler_dados() -> dict[str, int]:
        from triagem.config import Settings
        from triagem.retreino import carregar_dados

        treino, teste = carregar_dados(Settings.from_env())
        return {"n_treino": len(treino), "n_teste": len(teste)}

    @task
    def treinar(_resumo_dados: dict[str, int]) -> str:  # o argumento so cria a dependencia
        from triagem.config import Settings
        from triagem.retreino import carregar_dados, treinar_em_staging

        settings = Settings.from_env()
        treino, teste = carregar_dados(settings)
        return str(treinar_em_staging(settings, treino, teste))

    @task
    def validar_modelo(staging_dir: str, params: dict | None = None) -> dict:
        from pathlib import Path

        from triagem.config import Settings
        from triagem.retreino import validar_staging

        params = params or {}
        return validar_staging(
            Settings.from_env(),
            Path(staging_dir),
            f1_minimo=float(params.get("f1_minimo", F1_URGENCIA_MINIMO)),
            recall_minimo=float(params.get("recall_minimo", RECALL_URGENTE_MINIMO)),
        )

    @task
    def publicar_modelo(staging_dir: str, validacao: dict) -> str:
        from pathlib import Path

        from triagem.config import Settings
        from triagem.retreino import promover_staging

        promover_staging(Settings.from_env(), Path(staging_dir))
        return validacao["versao"]

    staging = treinar(ler_dados())
    publicar_modelo(staging, validar_modelo(staging))


treino_triagem()
