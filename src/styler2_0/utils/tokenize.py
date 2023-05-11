import os
import re
from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path
from typing import Any

from antlr4 import CommonTokenStream, InputStream
from streamerate import stream

from src.antlr.JavaLexer import JavaLexer
from styler2_0.utils.checkstyle import CheckstyleFileReport

#######################################################################################
# DISCLAIMER!
# The preprocessing is not entirely identical because in the Styler preprocessing, 2
# bugs were identified:
#
# 1. Whitespaces before line breaks were simply omitted.
#
# 2. If a whitespace contained tabs and spaces, it was encoded only with the first
#   type and the second one was simply omitted.
#######################################################################################


@total_ordering
class ProcessedToken:
    """
    Base class for a processed token coming from the lexer.
    """

    def __init__(self, text: str, line: int, column: int) -> None:
        self.text = text
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return self.de_tokenize()

    def de_tokenize(self) -> str:
        """
        Return none tokenized representation.
        :return: Returns none tokenized representation.
        """
        return self.text

    def __lt__(self, other: Any) -> bool:
        assert isinstance(other, ProcessedToken)
        if self.line == other.line:
            return self.column < other.column
        return self.line < other.line

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ProcessedToken):
            return False
        return (
            self.text == other.text
            and self.line == other.line
            and self.column == other.column
        )


class Identifier(ProcessedToken):
    """
    Class which represents Identifier/Literals
    """

    def __str__(self) -> str:
        return "IDENTIFIER"


class Whitespace(ProcessedToken):
    """
    Class which represents a whitespace.
    """

    def __init__(self, text: str, line: int, column: int, indent: str = ""):
        super().__init__(text, line, column)
        self.indent = indent

    def __str__(self) -> str:
        if self.text == "":
            return "0_None"
        return "_".join(
            self._process_whitespace_type(group)
            for group in re.findall(" +|\n+|\t+", self.text)
        )

    def is_linebreak(self):
        """
        Checks whether the current whitespace is a linebreak or not.
        :return: Returns True if it is a linebreak, otherwise False.
        """
        return "\n" in self.text

    def _process_whitespace_type(self, whitespace: str) -> str:
        match list(whitespace):
            case ["\n", *_]:
                return (
                    f"{len(whitespace)}_NL{f'_{self.indent}' * int(bool(self.indent))}"
                )
            case ["\t", *_]:
                return f"{len(whitespace)}_TB"
            case [" ", *_]:
                return f"{len(whitespace)}_SP"
            case _:
                raise ValueError("There was a non-whitespace char passed.")


class CheckstyleToken(ProcessedToken):
    """
    Token representing a checkstyle violation.
    """

    def __init__(self, text: str, line: int, column: int, is_starting: bool) -> None:
        super().__init__(text, line, column)
        self.is_starting = is_starting

    def __str__(self) -> str:
        return f"<{'/' * int(not self.is_starting)}{self.text}>"


class ProcessedSourceFile:
    """
    Class which represents a processed/tokenized Java file.
    """

    def __init__(
        self,
        file_name: Path,
        tokens: list[ProcessedToken],
        report: CheckstyleFileReport = None,
    ) -> None:
        self.file_name = file_name
        self.tokens = tokens

        # remove the eof if presents in tokens.
        self._remove_eof()

        # places where possible placeholder for whitespaces.
        self._insert_placeholder_ws()

        # insert deltas of indentation after linebreak
        self._insert_deltas_on_linebreak()

        if report:
            self._insert_checkstyle_report(report)

    def de_tokenize(self) -> str:
        """
        Get the real Java source code.
        :return: The original Java source code.
        """
        return "".join([token.de_tokenize() for token in self.tokens])

    def tokenized_str(self) -> str:
        """
        Get the tokenized/processed string of the Java file.
        :return: The processed string representation of the Java file.
        """
        return f"{' '.join(map(str, self.tokens))}\n"

    def __repr__(self) -> str:
        return self.tokenized_str()

    def _insert_placeholder_ws(self) -> None:
        padded_tokens = []
        for token, suc in zip(self.tokens[:-1], self.tokens[1:], strict=True):
            padded_tokens.append(token)
            if not (isinstance(token, Whitespace) or isinstance(suc, Whitespace)):
                padded_tokens.append(Whitespace("", token.line, suc.column))
        padded_tokens.append(self.tokens[-1])
        last_token = padded_tokens[-1]
        if not isinstance(last_token, Whitespace):
            padded_tokens.append(
                Whitespace(
                    "",
                    last_token.line,
                    last_token.column + len(last_token.de_tokenize()),
                )
            )
        self.tokens = padded_tokens

    def _remove_eof(self):
        if self.tokens[-1].text == "<EOF>":
            self.tokens = self.tokens[:-1]

    def _insert_checkstyle_report(self, report: CheckstyleFileReport) -> None:
        for violation in report.violations:
            start_ctx = (
                stream(self.tokens)
                .filter(lambda token, vl=violation.line: token.line == vl)
                .next()
            )
            end_ctx = (
                stream(self.tokens)
                .filter(lambda token, vl=violation.line: token.line > vl)
                .next()
            )
            start_style_token = CheckstyleToken(
                violation.type.value, start_ctx.line, start_ctx.column, True
            )
            end_style_token = CheckstyleToken(
                violation.type.value, start_ctx.line, start_ctx.column, False
            )
            start_ctx_idx = self.tokens.index(start_ctx)
            end_ctx_idx = self.tokens.index(end_ctx)
            self.tokens = (
                self.tokens[:start_ctx_idx]
                + [start_style_token]
                + self.tokens[start_ctx_idx:end_ctx_idx]
                + [end_style_token]
                + self.tokens[end_ctx_idx:]
            )

    def _insert_deltas_on_linebreak(self) -> None:
        linebreaks = (
            stream(self.tokens)
            .filter(lambda token: isinstance(token, Whitespace))
            .filter(lambda ws: ws.is_linebreak())
            .to_list()
        )
        for linebreak in linebreaks:
            this_line = linebreak.line
            indent_this_line = self._get_first_non_ws_token_of_line(this_line).column
            indent_next_line = self._get_first_non_ws_token_of_line(
                min(self.tokens[-1].line, this_line + 1)
            ).column
            delta = indent_next_line - indent_this_line
            print(delta)
            if delta < 0:
                linebreak.indent = "DD"
            if delta > 0:
                linebreak.indent = "ID"

    def _get_first_non_ws_token_of_line(self, line: int) -> ProcessedToken:
        return next(
            (
                token
                for token in self.tokens
                if token.line == line and not isinstance(token, Whitespace)
            ),
            None,
        )


class ContainsStr(str):
    """
    Helper class inheriting from str to allow match expression with contains on str.
    """

    def __eq__(self, other):
        return self.__contains__(other)


@dataclass(eq=True, frozen=True)
class RawToken:
    """
    Raw token passed from the lexer.
    """

    symbolic_name: str
    text: str
    line: int
    column: int

    def process(self) -> ProcessedToken:
        """
        Turn the token into a ProcessedToken which can than be used in further actions.
        :return: Returns the "processed" RawToken.
        """
        match ContainsStr(self.symbolic_name):
            case "IDENTIFIER" | "LITERAL":
                return Identifier(self.text, self.line, self.column)
            case "WS":
                return Whitespace(self.text, self.line, self.column)
            case _:
                return ProcessedToken(self.text, self.line, self.column)


def _tokenize_java_code(code: str) -> list[ProcessedToken]:
    input_stream = InputStream(code)
    lexer = JavaLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    token_stream.fill()
    return [
        RawToken(
            lexer.symbolicNames[token.type], token.text, token.line, token.column
        ).process()
        for token in token_stream.tokens
    ]


def tokenize_dir(directory: Path) -> list[ProcessedSourceFile]:
    """
    Parses the Java files form a given directory into a list of ProcessedSourceFile.
    :param directory: The given directory
    :return: Returns the list of ProcessedSourceFile.
    """
    processed_java_files = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".java"):
                continue
            with open(Path(subdir) / Path(file), encoding="utf-8") as file_stream:
                content = file_stream.read()
                tokens = _tokenize_java_code(content)
                processed_java_file = ProcessedSourceFile(
                    Path(subdir) / Path(file), tokens
                )
                processed_java_files.append(processed_java_file)
    return processed_java_files


def tokenize_with_reports(
    reports: frozenset[CheckstyleFileReport],
) -> list[ProcessedSourceFile]:
    """
    Tokenize a given checkstyle report.
    :param reports: The given checkstyle report.
    :return: Returns the tokenized source files with the inserted reports.
    """
    processed_files = []
    for report in reports:
        with open(report.path, encoding="utf-8") as source_file:
            if not source_file.name.endswith(".java"):
                continue
            content = source_file.read()
            tokens = _tokenize_java_code(content)
            processed_file = ProcessedSourceFile(report.path, tokens, report)
            processed_files.append(processed_file)
    return processed_files
