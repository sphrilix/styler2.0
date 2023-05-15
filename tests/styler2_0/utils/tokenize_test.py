from pathlib import Path

from src.styler2_0.utils.tokenize import (
    ProcessedSourceFile,
    Whitespace,
    tokenize_java_code,
)


def test_pad_source_file() -> None:
    tokens = tokenize_java_code(
        'public class Main {void test() {System.out.println("test");}}'
    )
    processed_file = ProcessedSourceFile(Path("."), tokens)
    assert len(processed_file.tokens) == 40
    assert isinstance(processed_file.tokens[-1], Whitespace)


def test_parse_whitespace() -> None:
    ws = Whitespace("\t\t\n  \n\n\t  ", 0, 0)
    assert str(ws) == "2_TB_1_NL_2_SP_2_NL_1_TB_2_SP"
