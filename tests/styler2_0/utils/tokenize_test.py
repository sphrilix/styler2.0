import os
from pathlib import Path

import pytest

from src.styler2_0.utils.checkstyle import (
    CheckstyleFileReport,
    Violation,
    ViolationType,
    run_checkstyle_on_str,
)
from src.styler2_0.utils.tokenize import (
    Comment,
    ProcessedSourceFile,
    Whitespace,
    tokenize_java_code,
)

CURR_DIR = os.path.dirname(os.path.relpath(__file__))
SAMPLE_PROJECT = os.path.join(CURR_DIR, "../../res/sample_project")
CHECKSTYLE_CONFIG = Path(os.path.join(SAMPLE_PROJECT, "checkstyle.xml"))


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
    report = CheckstyleFileReport(Path("."), frozenset(violation))
    tokens = tokenize_java_code("class Main {}")
    source = ProcessedSourceFile(Path("."), tokens, report)
    assert source.tokens[0].text == "FileTabCharacter"
    assert source.tokens[4].text == "FileTabCharacter"


def test_tokenizing_with_report_end() -> None:
    tokens = tokenize_java_code("class Main {}\n")
    violation = [
        Violation(ViolationType.FILE_TAB_CHARACTER, tokens[-1].line, tokens[-1].column)
    ]
    report = CheckstyleFileReport(Path("."), frozenset(violation))
    source = ProcessedSourceFile(Path("."), tokens, report)
    assert source.tokens[4].text == "FileTabCharacter"
    assert source.tokens[-1].text == "FileTabCharacter"


def test_tokenizing_with_line_violation() -> None:
    tokens = tokenize_java_code("class Main {}")
    violation = [Violation(ViolationType.REGEXP_SINGLE_LINE, 1, None)]
    report = CheckstyleFileReport(Path("."), frozenset(violation))
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
    report = CheckstyleFileReport(Path("."), frozenset(violation))
    source = ProcessedSourceFile(Path("."), tokens, report)
    assert source.tokens[4].text == ViolationType.REGEXP_SINGLE_LINE.value
    assert source.tokens[-2].text == ViolationType.REGEXP_SINGLE_LINE.value


def test_tokenizing_do_while_with_break() -> None:
    tokens = tokenize_java_code(
        "interface IntUtils {\n"
        "  @NotNull"
        "  default int foo(int param) {\n"
        "    do {\n"
        "      param = param - 42;\n"
        "      if (param < 0)\n"
        "        break;\n"
        "    } while(param > 1);\n"
        "  }\n"
        "}\n"
    )
    assert len(tokens) == 78
    assert len([t for t in tokens if str(t) == "int"]) == 2
    assert len([t for t in tokens if str(t) == "DECIMAL_LITERAL"]) == 3
    assert len([t for t in tokens if str(t) == "default"]) == 1
    assert len([t for t in tokens if str(t) == "while"]) == 1
    assert len([t for t in tokens if str(t) == "do"]) == 1
    assert len([t for t in tokens if str(t) == "interface"]) == 1


def test_literal_parsing() -> None:
    string = tokenize_java_code('"Hello"')
    assert str(string[0]) == "STRING_LITERAL"
    null = tokenize_java_code("null")
    assert str(null[0]) == "NULL_LITERAL"
    integer = tokenize_java_code("42")
    assert str(integer[0]) == "DECIMAL_LITERAL"
    hexa = tokenize_java_code("0x42")
    assert str(hexa[0]) == "HEX_LITERAL"
    hexa_fp = tokenize_java_code("0x1p3")
    assert str(hexa_fp[0]) == "HEX_FLOAT_LITERAL"
    boolean = tokenize_java_code("true")
    assert str(boolean[0]) == "BOOL_LITERAL"
    double = tokenize_java_code("42.0")
    assert str(double[0]) == "FLOAT_LITERAL"


def test_tokenizing_with_checkstyle_violation() -> None:
    violated_snippet = "public\tclass Violated{}"
    report = run_checkstyle_on_str(violated_snippet, "8.0", CHECKSTYLE_CONFIG)
    tokens = tokenize_java_code(violated_snippet)
    processed_source = ProcessedSourceFile(None, tokens, report)
    assert processed_source.tokens[0].text == "FileTabCharacter"
    assert processed_source.tokens[4].text == "FileTabCharacter"


def test_tokenizing_with_checkstyle_violation_nl() -> None:
    violated_snippet = (
        "public class Violated {\n        public\tvoid main(String[] args) {} }"
    )
    report = run_checkstyle_on_str(violated_snippet, "8.0", CHECKSTYLE_CONFIG)
    tokens = tokenize_java_code(violated_snippet)
    processed_source = ProcessedSourceFile(None, tokens, report)
    assert processed_source.tokens[8].text == "FileTabCharacter"
    assert processed_source.tokens[12].text == "FileTabCharacter"


def test_tokenize_with_line_violations() -> None:
    violated_snippet = (
        "public class Violated {\n"
        "    public void main(String[] args) {\n"
        "        new RuntimeException().printStackTrace();\n"
        "    }\n"
        "}"
    )
    report = run_checkstyle_on_str(violated_snippet, "8.0", CHECKSTYLE_CONFIG)
    tokens = tokenize_java_code(violated_snippet)
    processed_source = ProcessedSourceFile(None, tokens, report)
    assert processed_source.tokens[26].text == "RegexpSinglelineJava"
    assert processed_source.tokens[48].text == "RegexpSinglelineJava"


def _empty_class() -> str:
    return "public class Empty { %s }"


def _empty_class_with_missing_variable_name() -> str:
    return _empty_class() % "public int %s = 42;"


def _identifier_test_cases() -> list[tuple[str, str]]:
    return [
        (
            _empty_class_with_missing_variable_name() % "camelCase",
            "[I_LOWER] [I_FIRST_UPPER_OTHER_LOWER]",
        ),
        (
            _empty_class_with_missing_variable_name() % "snake_case",
            "[I_LOWER] [I_UNDERSCORE] [I_LOWER]",
        ),
        (
            _empty_class_with_missing_variable_name() % "CONST_CASE",
            "[I_UPPER] [I_UNDERSCORE] [I_UPPER]",
        ),
        (_empty_class_with_missing_variable_name() % "number42", "[I_LOWER]"),
        (
            _empty_class_with_missing_variable_name() % "snake_Case",
            "[I_LOWER] [I_UNDERSCORE] [I_FIRST_UPPER_OTHER_LOWER]",
        ),
        (
            _empty_class_with_missing_variable_name() % "CONST_Case",
            "[I_UPPER] [I_UNDERSCORE] [I_FIRST_UPPER_OTHER_LOWER]",
        ),
    ]


@pytest.mark.parametrize("inp, expected", _identifier_test_cases())  # noqa: PT006
def test_tokenize_identifiers(inp: str, expected: str) -> None:
    tokens = tokenize_java_code(inp)
    assert str(tokens[12]) == expected
