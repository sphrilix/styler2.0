from pathlib import Path

from src.styler2_0.utils.checkstyle import CheckstyleReport, Violation, ViolationType
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


def test_tokenizing_with_report_begin() -> None:
    violation = [Violation(ViolationType.FILE_TAB_CHARACTER, 1, 0)]
    report = CheckstyleReport(Path("."), frozenset(violation))
    tokens = tokenize_java_code("class Main {}")
    source = ProcessedSourceFile(Path("."), tokens, report)
    assert source.tokens[0].text == "FileTabCharacter"
    assert source.tokens[4].text == "FileTabCharacter"


def test_tokenizing_with_report_end() -> None:
    tokens = tokenize_java_code("class Main {}\n")
    violation = [
        Violation(ViolationType.FILE_TAB_CHARACTER, tokens[-1].line, tokens[-1].column)
    ]
    report = CheckstyleReport(Path("."), frozenset(violation))
    source = ProcessedSourceFile(Path("."), tokens, report)
    assert source.tokens[4].text == "FileTabCharacter"
    assert source.tokens[-1].text == "FileTabCharacter"


def test_tokenizing_with_line_violation() -> None:
    tokens = tokenize_java_code("class Main {}")
    violation = [Violation(ViolationType.REGEXP_SINGLE_LINE, 1, None)]
    report = CheckstyleReport(Path("."), frozenset(violation))
    source = ProcessedSourceFile(Path("."), tokens, report)
    assert source.tokens[0].text == ViolationType.REGEXP_SINGLE_LINE.value
    assert source.tokens[-1].text == ViolationType.REGEXP_SINGLE_LINE.value


def test_tokenizing_with_line_violation_in_between() -> None:
    tokens = tokenize_java_code(
        "class Main {\n"
        "    public static void main(String[] args) {"
        '        System.out.println("Hello World");\n'
        "    }"
        "}\n"
    )
    violation = [Violation(ViolationType.REGEXP_SINGLE_LINE, 2, None)]
    report = CheckstyleReport(Path("."), frozenset(violation))
    source = ProcessedSourceFile(Path("."), tokens, report)
    assert source.tokens[4].text == ViolationType.REGEXP_SINGLE_LINE.value
    assert source.tokens[-2].text == ViolationType.REGEXP_SINGLE_LINE.value
