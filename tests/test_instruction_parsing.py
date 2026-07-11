from reasoning.instruction_parsing import extract_target_color


def test_extract_target_color_english():
    assert extract_target_color("pick up the blue cube") == "blue"


def test_extract_target_color_portuguese():
    assert extract_target_color("pegue o cubo vermelho e mova") == "red"


def test_extract_target_color_is_case_insensitive():
    assert extract_target_color("PEGUE O CUBO AZUL") == "blue"


def test_extract_target_color_returns_none_when_no_color_mentioned():
    assert extract_target_color("faça alguma coisa qualquer") is None
