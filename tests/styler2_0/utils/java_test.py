import pytest
from streamerate import stream

from styler2_0.utils.java import NonParseableException, lex_java, returns_valid_java


def test_returns_in_fact_valid_java() -> None:
    assert (
        returns_valid_java(lambda: "public class Java {}")() == "public class Java {}"
    )


def test_returns_invalid_java() -> None:
    with pytest.raises(NonParseableException):
        returns_valid_java(lambda: "public class String int;")()


def test_lex_java() -> None:
    tokens = lex_java("public class Java {}")
    assert len(tokens) == 8
    assert (
        len(stream(tokens).filter(lambda token: token.symbolic_name == "WS").to_list())
        == 3
    )


def test_lex_invalid_java() -> None:
    with pytest.raises(NonParseableException):
        lex_java("public class String int;")
