"""Testes do benchmark de latencia: estatisticas, medicao, cliente HTTP e relatorio."""

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from triagem.benchmark import (
    ClienteHttp,
    Estatisticas,
    Parametros,
    RespostaHttpInvalidaError,
    amostrar_textos,
    calcular_estatisticas,
    executar,
    formatar_tabela,
    ler_resposta,
    main,
    medir,
)
from triagem.classificador import Classificador
from triagem.config import Settings

TEXTOS = ["cardiac aortic coronary", "tumor carcinoma neoplasm", "neuron cerebral seizure"]
VERSAO_FALSA = "falso@2026-01-01T00:00:00+00:00"


class _ServidorFalso(ThreadingHTTPServer):
    """API falsa: responde /health e /predict, com status configuravel."""

    status_predict = 200
    encerrar_conexao_apos_health = False


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # conexao persistente, como a API real

    def _responder(self, status: int, corpo: dict[str, str]) -> None:
        dados = json.dumps(corpo).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def do_GET(self) -> None:  # noqa: N802
        self._responder(200, {"status": "ok", "versao_modelo": VERSAO_FALSA})
        if getattr(self.server, "encerrar_conexao_apos_health", False):
            # Simula o timeout de keep-alive do servidor: fecha a conexao ociosa sem avisar
            # com "Connection: close", como o uvicorn faz apos 5 s sem requisicoes.
            self.close_connection = True

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        status = getattr(self.server, "status_predict", 200)
        self._responder(status, {"urgencia": "urgente"})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silencia o log do servidor de teste."""


@pytest.fixture
def servidor() -> Iterator[_ServidorFalso]:
    servidor = _ServidorFalso(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=servidor.serve_forever, daemon=True)
    thread.start()
    yield servidor
    servidor.shutdown()
    servidor.server_close()


def _url(servidor: _ServidorFalso) -> str:
    return f"http://127.0.0.1:{servidor.server_address[1]}"


def test_estatisticas_de_serie_conhecida() -> None:
    e = calcular_estatisticas([float(i) for i in range(1, 101)])

    assert e.medicoes == 100
    assert e.media_ms == 50.5
    assert e.p50_ms == 50.5
    assert e.p95_ms == 95.05
    assert e.p99_ms == 99.01
    assert e.maximo_ms == 100.0


def test_estatisticas_rejeitam_serie_vazia() -> None:
    with pytest.raises(ValueError, match="Nenhuma latencia"):
        calcular_estatisticas([])


def test_amostragem_e_repetivel_e_sem_reposicao() -> None:
    textos = [f"laudo {i}" for i in range(50)]

    a = amostrar_textos(textos, 10, semente=1)
    b = amostrar_textos(textos, 10, semente=1)
    c = amostrar_textos(textos, 10, semente=2)

    assert a == b
    assert a != c
    assert len(set(a)) == 10


@pytest.mark.parametrize("quantidade", [0, -1, 51])
def test_amostragem_rejeita_quantidade_invalida(quantidade: int) -> None:
    with pytest.raises(ValueError, match="quantidade"):
        amostrar_textos([f"laudo {i}" for i in range(50)], quantidade, semente=1)


def test_medir_descarta_aquecimento_e_percorre_textos_em_ciclo() -> None:
    chamadas: list[str] = []

    latencias = medir(chamadas.append, ["a", "b"], aquecimento=3, medicoes=4)

    assert len(latencias) == 4
    assert all(latencia >= 0 for latencia in latencias)
    assert chamadas == ["a", "b", "a", "a", "b", "a", "b"]  # 3 de aquecimento + 4 medidas


@pytest.mark.parametrize(
    ("textos", "aquecimento", "medicoes"),
    [([], 1, 1), (["a"], -1, 1), (["a"], 0, 0)],
)
def test_medir_valida_argumentos(textos: list[str], aquecimento: int, medicoes: int) -> None:
    with pytest.raises(ValueError, match=r"texto|aquecimento"):
        medir(lambda _: None, textos, aquecimento, medicoes)


@pytest.mark.parametrize("url", ["https://localhost:8000", "localhost:8000", "http://"])
def test_cliente_rejeita_url_invalida(url: str) -> None:
    with pytest.raises(ValueError, match="http://host:porta"):
        ClienteHttp(url)


def test_cliente_http_le_versao_e_classifica_com_conexao_persistente(
    servidor: _ServidorFalso,
) -> None:
    cliente = ClienteHttp(_url(servidor))
    try:
        assert cliente.versao_modelo() == VERSAO_FALSA
        for texto in TEXTOS:  # varias chamadas na mesma conexao
            cliente.classificar(texto)
    finally:
        cliente.fechar()


def test_cliente_http_falha_com_status_diferente_de_200(servidor: _ServidorFalso) -> None:
    servidor.status_predict = 500
    cliente = ClienteHttp(_url(servidor))
    try:
        with pytest.raises(RespostaHttpInvalidaError, match="500"):
            cliente.classificar("texto")
    finally:
        cliente.fechar()


class _SoqueteGravador:
    """Envolve um socket real e registra cada `sendall` e cada `setsockopt`."""

    def __init__(self, real: socket.socket) -> None:
        self.real = real
        self.escritas: list[bytes] = []
        self.opcoes: list[tuple[Any, ...]] = []

    def sendall(self, dados: bytes) -> None:
        self.escritas.append(dados)
        self.real.sendall(dados)

    def setsockopt(self, *args: Any) -> None:
        self.opcoes.append(args)
        self.real.setsockopt(*args)

    def __getattr__(self, nome: str) -> object:
        return getattr(self.real, nome)


def test_cliente_envia_cabecalhos_e_corpo_numa_unica_escrita(
    servidor: _ServidorFalso, monkeypatch: pytest.MonkeyPatch
) -> None:
    gravadores: list[_SoqueteGravador] = []
    criar_conexao = socket.create_connection

    def criar_gravador(*args: Any, **kwargs: Any) -> _SoqueteGravador:
        gravador = _SoqueteGravador(criar_conexao(*args, **kwargs))
        gravadores.append(gravador)
        return gravador

    monkeypatch.setattr("triagem.benchmark.socket.create_connection", criar_gravador)
    cliente = ClienteHttp(_url(servidor))
    try:
        cliente.classificar("texto do laudo")
    finally:
        cliente.fechar()

    (gravador,) = gravadores
    assert len(gravador.escritas) == 1  # cabecalhos e corpo juntos, sem Nagle entre os dois
    assert b"Content-Length: " in gravador.escritas[0]
    assert gravador.escritas[0].endswith(b'{"texto": "texto do laudo"}')
    assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in gravador.opcoes


def test_ler_resposta_junta_cabecalho_e_corpo_recebidos_em_partes() -> None:
    cliente, servidor_ = socket.socketpair()
    with cliente, servidor_:
        servidor_.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\nhello")
        servidor_.sendall(b" world")

        status, corpo = ler_resposta(cliente)

    assert (status, corpo) == (200, b"hello world")


def test_ler_resposta_exige_content_length() -> None:
    cliente, servidor_ = socket.socketpair()
    with cliente, servidor_:
        servidor_.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")

        with pytest.raises(RespostaHttpInvalidaError, match="Content-Length"):
            ler_resposta(cliente)


def test_ler_resposta_falha_se_o_servidor_encerrar_a_conexao() -> None:
    cliente, servidor_ = socket.socketpair()
    with cliente:
        servidor_.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 50\r\n\r\ncurto")
        servidor_.close()

        with pytest.raises(ConnectionError):
            ler_resposta(cliente)


def test_executar_so_inferencia(settings_com_modelo: Settings) -> None:
    classificador = Classificador.carregar(settings_com_modelo)
    parametros = Parametros(amostras=3, aquecimento=2, medicoes=5, semente=1)

    relatorio = executar(classificador, TEXTOS, None, "teste", parametros)

    assert relatorio["rotulo"] == "teste"
    assert relatorio["parametros"] == {
        "amostras": 3,
        "aquecimento": 2,
        "medicoes": 5,
        "semente": 1,
    }
    assert set(relatorio["resultados"]) == {"inferencia"}
    assert relatorio["resultados"]["inferencia"]["medicoes"] == 5
    assert relatorio["modelo"] == {"versao_local": classificador.versao, "versao_http": None}
    assert {"python", "plataforma", "cpus", "scikit_learn", "numpy"} <= set(relatorio["ambiente"])


def test_executar_com_http_registra_versao_da_api(
    settings_com_modelo: Settings, servidor: _ServidorFalso
) -> None:
    classificador = Classificador.carregar(settings_com_modelo)
    parametros = Parametros(amostras=3, aquecimento=1, medicoes=4, semente=1)

    relatorio = executar(classificador, TEXTOS, _url(servidor), "teste", parametros)

    assert set(relatorio["resultados"]) == {"inferencia", "http"}
    assert relatorio["resultados"]["http"]["medicoes"] == 4
    assert relatorio["modelo"]["versao_http"] == VERSAO_FALSA


def test_executar_avisa_quando_modelo_da_api_difere(
    settings_com_modelo: Settings,
    servidor: _ServidorFalso,
    caplog: pytest.LogCaptureFixture,
) -> None:
    classificador = Classificador.carregar(settings_com_modelo)
    parametros = Parametros(amostras=3, aquecimento=0, medicoes=1, semente=1)

    with caplog.at_level("WARNING"):
        executar(classificador, TEXTOS, _url(servidor), "teste", parametros)

    assert "difere do local" in caplog.text


def test_executar_http_sobrevive_a_conexao_encerrada_pelo_servidor(
    settings_com_modelo: Settings, servidor: _ServidorFalso
) -> None:
    # Regressao: a inferencia leva segundos e o servidor encerra a conexao ociosa aberta
    # pela consulta ao /health; a medicao HTTP precisa abrir uma conexao nova.
    servidor.encerrar_conexao_apos_health = True
    classificador = Classificador.carregar(settings_com_modelo)
    parametros = Parametros(amostras=3, aquecimento=1, medicoes=4, semente=1)

    relatorio = executar(classificador, TEXTOS, _url(servidor), "teste", parametros)

    assert relatorio["resultados"]["http"]["medicoes"] == 4


def test_executar_falha_rapido_se_a_api_estiver_fora_do_ar(
    settings_com_modelo: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    with socket.socket() as soquete:  # porta livre e fechada em seguida
        soquete.bind(("127.0.0.1", 0))
        porta = soquete.getsockname()[1]
    chamadas: list[str] = []
    classificador = Classificador.carregar(settings_com_modelo)
    monkeypatch.setattr(classificador, "classificar", chamadas.append)

    with pytest.raises(OSError):
        executar(classificador, TEXTOS, f"http://127.0.0.1:{porta}", "teste", Parametros())

    assert chamadas == []  # nao mediu a inferencia antes de descobrir que a API nao responde


def test_formatar_tabela_lista_cada_nivel() -> None:
    estatisticas = Estatisticas(
        medicoes=10, media_ms=1.5, p50_ms=1.2, p95_ms=2.5, p99_ms=3.5, maximo_ms=4.5
    )

    tabela = formatar_tabela({"inferencia": estatisticas, "http": estatisticas})

    assert "inferencia" in tabela
    assert "http" in tabela
    assert "P99" in tabela
    assert "3.50" in tabela


def _preparar_main(monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path) -> Path:
    """Aponta o ambiente para o modelo de teste e substitui o download por um CSV local."""
    monkeypatch.setenv("TRIAGEM_ARTIFACTS_DIR", str(settings.artifacts_dir))
    monkeypatch.setenv("TRIAGEM_DATA_DIR", str(tmp_path / "data"))
    csv_teste = tmp_path / "teste.csv"
    linhas = ["condition_label,medical_abstract"]
    linhas += [f"{(i % 5) + 1},{texto} caso {i}" for i, texto in enumerate(TEXTOS * 2)]
    csv_teste.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    monkeypatch.setattr("triagem.benchmark.baixar_arquivo", lambda *_args, **_kwargs: csv_teste)
    return tmp_path / "relatorio.json"


def test_main_grava_relatorio_json(
    monkeypatch: pytest.MonkeyPatch,
    settings_com_modelo: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    saida = _preparar_main(monkeypatch, settings_com_modelo, tmp_path)

    main(["--amostras", "3", "--aquecimento", "1", "--medicoes", "4", "--saida", str(saida)])

    relatorio = json.loads(saida.read_text(encoding="utf-8"))
    assert relatorio["resultados"]["inferencia"]["medicoes"] == 4
    assert relatorio["rotulo"] == "sklearn"
    assert "inferencia" in capsys.readouterr().out


def test_main_sai_com_erro_se_o_modelo_nao_existe(
    monkeypatch: pytest.MonkeyPatch, settings_com_modelo: Settings, tmp_path: Path
) -> None:
    saida = _preparar_main(monkeypatch, settings_com_modelo, tmp_path)
    monkeypatch.setenv("TRIAGEM_ARTIFACTS_DIR", str(tmp_path / "vazio"))

    with pytest.raises(SystemExit) as erro:
        main(["--amostras", "3", "--medicoes", "2", "--saida", str(saida)])

    assert erro.value.code == 1
    assert not saida.exists()
