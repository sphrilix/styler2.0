from pathlib import Path

from src.styler2_0.utils.tokenize import (
    Identifier,
    ProcessedSourceFile,
    ProcessedToken,
    Whitespace,
)


def test_pad_source_file() -> None:
    tokens: list[ProcessedToken] = [
        Identifier("Test", 0, 0),
        Identifier("Test2", 0, 4),
        Whitespace("", 0, 4),
    ]
    processed_file = ProcessedSourceFile(Path("."), tokens)
    assert len(processed_file.tokens) == 4
    assert isinstance(processed_file.tokens[1], Whitespace)
    assert isinstance(processed_file.tokens[3], Whitespace)


def test_parse_whitespace() -> None:
    ws = Whitespace("\t\t\n  \n\n\t  ", 0, 0)
    assert str(ws) == "2_TB_1_NL_2_SP_2_NL_1_TB_2_SP"
