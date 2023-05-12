import os
import re
from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path
from typing import Any

from antlr4 import CommonTokenStream, InputStream
from streamerate import stream

from src.antlr.JavaLexer import JavaLexer
from styler2_0.utils.checkstyle import CheckstyleFileReport, Violation

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


class Comment(ProcessedToken):
    """
    Class representing a comment token.
    """

    def __str__(self) -> str:
        return "COMMENT"


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

        self.non_ws_tokens = (
            stream(self.tokens)
            .filter(lambda token: not isinstance(token, Whitespace))
            .to_list()
        )

        # places where possible placeholder for whitespaces.
        self._insert_placeholder_ws()

        # insert deltas of indentation after linebreak
        self._insert_deltas_on_linebreaks()

        # if report is given insert it into token sequence
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
        assert report.path == self.file_name, "Report and source file path must match."

        for violation in report.violations:
            start_ctx, end_ctx = self._calc_ctx(violation)

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

    def _calc_ctx(
        self, violation: Violation, ctx_line: int = 6, ctx_around: int = 1
    ) -> (ProcessedToken, ProcessedToken):
        """
        Currently reimplementation of styler checkstyle context calculation.

        DISCLAIMER!
        Since the source code and the paper do not contain any specification how this
        is done, it cannot be guaranteed whether this is 100% correct. (Assuming it is
        not 100% identical)

        :param violation: Violation which should be inserted.
        :param ctx_line: How many lines should be taken into account.
        :param ctx_around: Amount of tokens around violation which should be taken
                           into account.
        :return: Returns the starting and ending token of the context.
        """
        (
            ctx_begin,
            ctx_begin_token_idx,
            ctx_end,
            ctx_end_token_idx,
        ) = self._calc_line_ctx(ctx_line, violation)

        violation_start_idx, violation_end_idx = self._calc_col_ctx(
            ctx_around,
            ctx_begin,
            ctx_begin_token_idx,
            ctx_end,
            ctx_end_token_idx,
            violation,
        )

        # normalize tokens according to indexes
        start_ctx_token = self.non_ws_tokens[max(0, violation_start_idx)]

        # if ctx length is bigger than non whitespace token set context end to
        # first whitespace after last non whitespace token
        if violation_end_idx == len(self.non_ws_tokens):
            end_ctx_token = self.tokens[
                self.tokens.index(self.non_ws_tokens[violation_end_idx - 1]) + 1
            ]
        else:
            end_ctx_token = self.non_ws_tokens[violation_end_idx]

        return (
            start_ctx_token,
            end_ctx_token,
        )

    def _calc_col_ctx(
        self,
        ctx_around: int,
        ctx_begin: int,
        ctx_begin_token_idx: int,
        ctx_end: int,
        ctx_end_token_idx: int,
        violation: Violation,
    ) -> (int, int):
        violation_ctx_idx = -1

        # if we encounter a violation with column information

        if violation.column:
            vl_col = violation.column
            if vl_col <= self.non_ws_tokens[ctx_begin].column:
                violation_ctx_idx = ctx_begin
            elif vl_col >= self.non_ws_tokens[ctx_end - 1].column:
                violation_ctx_idx = ctx_end - 1
            else:
                idx = ctx_begin_token_idx
                for token in self.non_ws_tokens[ctx_begin:ctx_end]:
                    if token.column <= vl_col:
                        violation_ctx_idx = idx
                    idx += 1
            violation_start_idx = max(0, violation_ctx_idx - ctx_around)
            violation_end_idx = min(
                len(self.non_ws_tokens), violation_ctx_idx + ctx_around
            )
        else:
            if ctx_begin > -1:
                violation_start_idx = max(0, ctx_begin_token_idx - ctx_around)
                violation_end_idx = min(
                    len(self.non_ws_tokens), ctx_end_token_idx + ctx_around
                )
            else:
                for idx, token in enumerate(self.non_ws_tokens):
                    if token.line < violation.line:
                        violation_ctx_idx = idx
                violation_start_idx = max(0, violation_ctx_idx - ctx_around)
                violation_end_idx = min(
                    len(self.non_ws_tokens), violation_ctx_idx + ctx_around
                )

        return violation_start_idx, violation_end_idx

    def _calc_line_ctx(
        self, ctx_line: int, violation: Violation
    ) -> (int, int, int, int):
        ctx_begin = len(self.non_ws_tokens)
        ctx_end = 0
        ctx_begin_token_idx = -1
        ctx_end_token_idx = -1
        token_start = False

        # calc starting and end token idx according to context.
        for idx, token in enumerate(self.non_ws_tokens):
            if violation.line - ctx_line <= token.line <= violation.line + ctx_line:
                ctx_begin_token_idx = min(idx, ctx_begin)
                ctx_end_token_idx = max(idx, ctx_end)
            if not token_start and violation.line == token.line:
                token_start = True
                ctx_begin = idx
            if token_start and violation.line < token.line:
                token_start = False
                ctx_end = idx

        # normalize start and end token idx
        ctx_begin_token_idx = max(0, ctx_begin_token_idx - 1)
        ctx_end_token_idx = min(len(self.non_ws_tokens), ctx_end_token_idx + 1)
        if ctx_end < 0:
            ctx_end_token_idx = ctx_begin_token_idx

        return (
            ctx_begin,
            ctx_begin_token_idx,
            ctx_end,
            ctx_end_token_idx,
        )

    def _insert_deltas_on_linebreaks(self) -> None:
        linebreaks = (
            stream(self.tokens)
            .filter(lambda token: isinstance(token, Whitespace))
            .filter(lambda ws: ws.is_linebreak())
            .to_list()
        )
        for linebreak in linebreaks:
            this_line = linebreak.line
            indent_this_line = self._calculate_indent_of_line(this_line)
            indent_next_line = self._calculate_indent_of_line(
                min(self.tokens[-1].line, this_line + 1)
            )
            delta = indent_next_line - indent_this_line
            if delta < 0:
                linebreak.indent = "DD"
            if delta > 0:
                linebreak.indent = "ID"

    def _calculate_indent_of_line(self, line: int) -> int:
        first_non_ws_token = self._get_first_non_ws_token_of_line(line)
        return first_non_ws_token.column if first_non_ws_token else 0

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
            case "COMMENT":
                return Comment(self.text, self.line, self.column)
            case _:
                return ProcessedToken(self.text, self.line, self.column)


def tokenize_java_code(code: str) -> list[ProcessedToken]:
    """
    Tokenize a given code snippet into ProcessedTokens.
    :param code: The given code snippet.
    :return: Returns the ProcessedTokens
    """
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
                tokens = tokenize_java_code(content)
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
            print(source_file)
            tokens = tokenize_java_code(content)
            processed_file = ProcessedSourceFile(report.path, tokens, report)
            processed_files.append(processed_file)
    return processed_files
