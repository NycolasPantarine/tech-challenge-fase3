from triagem.texto import normalizar


def test_converte_para_minusculas() -> None:
    assert normalizar("Heart FAILURE") == "heart failure"


def test_remove_stop_words_e_tokens_de_um_caractere() -> None:
    assert normalizar("The patient has a fever, and x") == "patient fever"


def test_texto_sem_tokens_uteis_vira_string_vazia() -> None:
    assert normalizar("the of and ...") == ""


def test_normalizacao_e_idempotente() -> None:
    uma_vez = normalizar("Acute Myocardial Infarction in the elderly")

    assert normalizar(uma_vez) == uma_vez
