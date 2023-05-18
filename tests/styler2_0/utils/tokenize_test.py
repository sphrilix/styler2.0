from pathlib import Path

from src.styler2_0.utils.tokenize import (
    Comment,
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


def test_ending_position() -> None:
    ws = Whitespace("\t\t\n  \n\n\t  ", 0, 0)
    end_line, end_col = ws.ending_position()
    assert end_line == 3
    assert end_col == 3
    ws_1_line = Whitespace("   ", 42, 42)
    end_1_line, end_1_col = ws_1_line.ending_position()
    assert end_1_line == ws_1_line.line
    assert end_1_col == 45
    comment = Comment("/*this is multiline\ncomment!*/", 0, 0)
    com_end_line, com_end_col = comment.ending_position()
    assert com_end_line == 1
    assert com_end_col == 10
