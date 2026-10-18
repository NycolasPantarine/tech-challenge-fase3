import json
import shutil
from pathlib import Path

import joblib
import pytest
from onnx import helper

from triagem.classificador import Classificador, ModeloIndisponivelError
from triagem.config import ENV_BACKEND, Backend, Settings
from triagem.modelo import TipoModelo, construir_pipeline
from triagem.modelo_onnx import PALAVRA_RESERVADA, ModeloOnnx, preparar_texto
from triagem.onnx_export import (
    ConversaoOnnxError,
    ParidadeOnnxError,
    _corrigir_sublinear_tf,
    converter,
    exportar_onnx,
    verificar_paridade,
)
from triagem.texto import normalizar

# Mesmo vocabulario do artefato sintetico de teste (conftest), uma frase por categoria.
PALAVRAS = [
    "tumor carcinoma neoplasm metastasis oncology",
    "gastric hepatic colon bowel pancreas",
    "neuron cerebral seizure dementia stroke",
    "cardiac aortic coronary ventricular infarction",
    "infection fever inflammation sepsis syndrome",
]
LAUDOS = [f"{palavras} caso {i}" for i, palavras in enumerate(PALAVRAS)] + [
    "tumor with cardiac involvement and fever",
    "gastric neuron coronary sepsis carcinoma",
]


@pytest.fixture(scope="module")
def settings_com_onnx(settings_com_modelo: Settings, tmp_path_factory: pytest.TempPathFactory):
    """Copia do artefato de teste com o ONNX exportado (nao altera o artefato da sessao)."""
    destino = tmp_path_factory.mktemp("com_onnx")
    shutil.copytree(settings_com_modelo.artifacts_dir, destino / "artifacts")
    settings = Settings(
        data_dir=destino / "data", artifacts_dir=destino / "artifacts", backend=Backend.ONNX
    )
    exportar_onnx(settings, LAUDOS)
    return settings


def _copia(settings: Settings, tmp_path: Path) -> Settings:
    shutil.copytree(settings.artifacts_dir, tmp_path / "artifacts")
    return Settings(
        data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts", backend=Backend.ONNX
    )


def test_exportacao_grava_onnx_e_registra_nos_metadados(settings_com_onnx: Settings) -> None:
    metadata = json.loads(settings_com_onnx.metadata_path.read_text(encoding="utf-8"))
    registro = metadata["onnx"]

    assert settings_com_onnx.onnx_path.exists()
    assert registro["treinado_em"] == metadata["treinado_em"]
    assert len(registro["sha256"]) == 64
    assert registro["paridade"]["urgencias_divergentes"] == 0
    assert not list(settings_com_onnx.artifacts_dir.glob("*.novo"))  # sem temporarios


def test_onnx_decide_como_o_modelo_original(
    settings_com_onnx: Settings, settings_com_modelo: Settings
) -> None:
    original = Classificador.carregar(settings_com_modelo)
    otimizado = Classificador.carregar(settings_com_onnx)

    for i, laudo in enumerate(LAUDOS):
        a, b = original.classificar(laudo), otimizado.classificar(laudo)
        assert a.urgencia == b.urgencia
        if i < len(PALAVRAS):  # os laudos mistos terminam em empate tecnico entre categorias
            assert a.categoria_id == b.categoria_id
        for nivel, probabilidade in a.probabilidades_urgencia.items():
            assert b.probabilidades_urgencia[nivel] == pytest.approx(probabilidade, abs=0.02)


def test_versao_do_modelo_onnx_indica_o_backend(settings_com_onnx: Settings) -> None:
    assert Classificador.carregar(settings_com_onnx).versao.endswith("+onnx")


def test_texto_com_acentos_da_o_mesmo_resultado_nos_dois_motores(
    settings_com_onnx: Settings, settings_com_modelo: Settings
) -> None:
    laudo = "cardiac résumé infarction naïve café aortic"

    a = Classificador.carregar(settings_com_modelo).classificar(laudo)
    b = Classificador.carregar(settings_com_onnx).classificar(laudo)

    assert a.urgencia == b.urgencia
    assert b.categoria_probabilidade == pytest.approx(a.categoria_probabilidade, abs=0.02)


def test_preparar_texto_so_troca_tokens_nao_ascii() -> None:
    assert preparar_texto("cardiac aortic") == "cardiac aortic"
    assert preparar_texto("cardiac résumé aortic") == f"cardiac {PALAVRA_RESERVADA} aortic"


def test_carregar_onnx_sem_exportar_orienta_a_exportar(
    settings_com_modelo: Settings, tmp_path: Path
) -> None:
    settings = _copia(settings_com_modelo, tmp_path)

    with pytest.raises(ModeloIndisponivelError, match="onnx_export"):
        Classificador.carregar(settings)


def test_onnx_de_outro_treino_e_recusado(settings_com_onnx: Settings, tmp_path: Path) -> None:
    settings = _copia(settings_com_onnx, tmp_path)
    metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    metadata["treinado_em"] = "2030-01-01T00:00:00+00:00"  # como um retreino sem reexportar
    settings.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ModeloIndisponivelError, match="outro treino"):
        Classificador.carregar(settings)


def test_onnx_adulterado_e_recusado(settings_com_onnx: Settings, tmp_path: Path) -> None:
    settings = _copia(settings_com_onnx, tmp_path)
    settings.onnx_path.write_bytes(settings.onnx_path.read_bytes() + b"\x00")

    with pytest.raises(ModeloIndisponivelError, match="difere do registrado"):
        Classificador.carregar(settings)


def test_onnx_corrompido_com_hash_valido_e_recusado(
    settings_com_onnx: Settings, tmp_path: Path
) -> None:
    settings = _copia(settings_com_onnx, tmp_path)
    settings.onnx_path.write_bytes(b"isto nao e um modelo onnx")
    metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    import hashlib

    metadata["onnx"]["sha256"] = hashlib.sha256(settings.onnx_path.read_bytes()).hexdigest()
    settings.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ModeloIndisponivelError, match="ilegivel"):
        Classificador.carregar(settings)


def test_portao_de_paridade_reprova_modelo_diferente(
    settings_com_onnx: Settings, settings_com_modelo: Settings
) -> None:
    outro = construir_pipeline(TipoModelo.NAIVE_BAYES)
    outro.fit([normalizar(t) for t in LAUDOS * 3], [1, 2, 3, 4, 5, 1, 2] * 3)
    joblib_original = joblib.load(settings_com_modelo.model_path)

    # O ONNX e do pipeline original: comparado com o pipeline de outro modelo, deve reprovar.
    with pytest.raises(ParidadeOnnxError):
        verificar_paridade(outro, settings_com_onnx.onnx_path, LAUDOS, limiar_urgente=0.5)
    # E aprova quando e o pipeline certo.
    relatorio = verificar_paridade(
        joblib_original, settings_com_onnx.onnx_path, LAUDOS, limiar_urgente=0.5
    )
    assert relatorio["urgencias_divergentes"] == 0


def test_conversao_recusa_vocabulario_com_a_palavra_reservada() -> None:
    pipeline = construir_pipeline(TipoModelo.NAIVE_BAYES)
    textos = [f"{PALAVRA_RESERVADA} {p}" for p in PALAVRAS] * 10
    pipeline.fit(textos, [1, 2, 3, 4, 5] * 10)

    with pytest.raises(ConversaoOnnxError, match="reservado"):
        converter(pipeline)


def test_correcao_do_sublinear_falha_se_o_grafo_nao_tem_o_trecho_esperado() -> None:
    grafo_vazio = helper.make_model(helper.make_graph([], "vazio", [], []))

    with pytest.raises(ConversaoOnnxError, match="log\\(contagem \\+ 1\\)"):
        _corrigir_sublinear_tf(grafo_vazio)


def test_modelo_onnx_recusa_arquivo_sem_a_saida_esperada(tmp_path: Path) -> None:
    from onnx import TensorProto

    entrada = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    saida = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    grafo = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [entrada], [saida])
    caminho = tmp_path / "outro.onnx"
    caminho.write_bytes(
        helper.make_model(
            grafo, ir_version=10, opset_imports=[helper.make_opsetid("", 17)]
        ).SerializeToString()
    )

    with pytest.raises(ValueError, match="sem a saida"):
        ModeloOnnx.abrir(caminho, [1])


def test_backend_padrao_e_sklearn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_BACKEND, raising=False)

    assert Settings.from_env().backend is Backend.SKLEARN


def test_backend_onnx_pela_variavel_de_ambiente(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_BACKEND, "onnx")

    assert Settings.from_env().backend is Backend.ONNX


def test_backend_desconhecido_falha_com_mensagem_clara(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_BACKEND, "tensorrt")

    with pytest.raises(ValueError, match="sklearn, onnx"):
        Settings.from_env()
