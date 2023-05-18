import os
import re
from pathlib import Path

from streamerate import stream

from src.styler2_0.utils.checkstyle import CheckstyleReport, Violation
from src.styler2_0.utils.java import Lexeme, lex_java

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

JAVA_OPERATORS = [
    "=",
    ">",
    "<",
    "!",
    "~",
    "?",
    ":",
    "==",
    "<=",
    ">=",
    "!=",
    "&&",
    "||",
    "++",
    "--",
    "+",
    "-",
    "*",
    "/",
    "&",
    "|",
    "^",
    "%",
    "+=",
    "-=",
    "*=",
    "/=",
    "&=",
    "|=",
    "^=",
    "%=",
    "<<=",
    ">>=",
    ">>>=",
    "->",
    "::",
]
PUNCTUATION = [".", ",", "(", ")", "[", "]", "{", "}", ";"]


class Token:
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

    def is_operator(self) -> bool:
        """

        :return:
        """
        return self.text in JAVA_OPERATORS

    def is_punctuation(self) -> bool:
        return self.text in PUNCTUATION

    def ending_position(self) -> (int, int):
        """
        Get the ending position of the current token.
        :return: Returns (line, column) of the ending position.
        """
        count_nl = self.text.count("\n")
        end_col = len(self.text.split("\n")[-1])
        if not count_nl:
            end_col += self.column
        return self.line + count_nl, end_col

    def is_triggering(self, violation: Violation) -> bool:
        """
        Determines whether the token is triggering the given violation.
        :param violation: The given violation.
        :return: Returns True if is triggered by this token else False.
        """
        if not violation.column:
            return self.line == violation.line
        end_line, end_col = self.ending_position()
        if self.line < violation.line < end_line:
            return True
        if self.line == end_line:
            return (
                violation.line == self.line
                and self.column <= violation.column <= end_col
            )
        if violation.line == self.line:
            return self.column <= violation.column
        if violation.line == end_line:
            return end_col <= violation.column
        return False


class Identifier(Token):
    """
    Class which represents Identifier/Literals
    """

    def __str__(self) -> str:
        return "IDENTIFIER"


class Whitespace(Token):
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


class Comment(Token):
    """
    Class representing a comment token.
    """

    def __str__(self) -> str:
        return "COMMENT"


class CheckstyleToken(Token):
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
        tokens: list[Token],
        report: CheckstyleReport = None,
    ) -> None:
        self.file_name = file_name
        self.tokens = tokens

        self.non_ws_tokens = (
            stream(self.tokens)
            .filter(lambda token: not isinstance(token, Whitespace))
            .to_list()
        )

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

    def _insert_checkstyle_report(self, report: CheckstyleReport) -> None:
        assert report.path == self.file_name, "Report and source file path must match."

        for violation in report.violations:
            print(violation)
            if violation.column is None:
                start, end = self._get_line_violation_ctx(violation)
            else:
                start, end = self._get_col_violation_ctx(violation)

            start_style_token = CheckstyleToken(
                violation.type.value, start.line, start.column, True
            )
            end_style_token = CheckstyleToken(
                violation.type.value, end.line, end.column, False
            )
            start_idx = max(0, self.tokens.index(start))
            end_idx = min(len(self.tokens), self.tokens.index(end) + 1)
            self.tokens.insert(end_idx, end_style_token)
            self.tokens.insert(start_idx, start_style_token)

    def _get_line_violation_ctx(self, violation: Violation) -> (Token, Token):
        assert violation.column is None
        if violation.line == 0:
            start = self.tokens[0]
        else:
            start = self._get_last_non_ws_token_of_line(violation.line - 1)

        if violation.line == self.tokens[-1].line:
            end = self.tokens[-1]
        else:
            end = self._get_last_non_ws_token_of_line(violation.line + 1)

        return start, end

    def _get_col_violation_ctx(self, violation: Violation) -> (Token, Token):
        assert violation.column is not None
        affected_token = (
            stream(self.tokens)
            .filter(lambda token: token.is_triggering(violation))
            .next()
        )
        if affected_token == self.tokens[0]:
            start = self.tokens[0]
        else:
            start = self._get_next_non_ws_token(affected_token, True)

        if affected_token == self.tokens[-1]:
            end = self.tokens[-1]
        else:
            end = self._get_next_non_ws_token(affected_token, False)

        return start, end

    def _get_next_non_ws_token(self, token: Token, reverse: bool = False) -> Token:
        if reverse:
            return next(
                (
                    non_ws_token
                    for non_ws_token in reversed(self.non_ws_tokens)
                    if non_ws_token.line == token.line
                    and non_ws_token.column < token.column
                    or non_ws_token.line < token.line
                ),
                self.tokens[-1],
            )
        return next(
            (
                non_ws_token
                for non_ws_token in self.non_ws_tokens
                if non_ws_token.line == token.line
                and token.column < non_ws_token.column
                or token.line < non_ws_token.line
            ),
            self.tokens[0],
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

    def _get_first_non_ws_token_of_line(self, line: int) -> Token:
        return next(
            (token for token in self.non_ws_tokens if token.line == line),
            None,
        )

    def _get_last_non_ws_token_of_line(self, line: int) -> Token:
        return (
            stream(self.non_ws_tokens)
            .reversed()
            .filter(lambda token: token.line <= line)
            .next()
        )


class ContainsStr(str):
    """
    Helper class inheriting from str to allow match expression with contains on str.
    """

    def __eq__(self, other):
        return self.__contains__(other)


def _process_raw_token(raw_token: Lexeme) -> Token:
    """
    Turn the token into a ProcessedToken which can than be used in further actions.
    :return: Returns the "processed" RawToken.
    """
    match ContainsStr(raw_token.symbolic_name):
        case "IDENTIFIER" | "LITERAL":
            return Identifier(raw_token.text, raw_token.line, raw_token.column)
        case "WS":
            return Whitespace(raw_token.text, raw_token.line, raw_token.column)
        case "COMMENT":
            return Comment(raw_token.text, raw_token.line, raw_token.column)
        case _:
            return Token(raw_token.text, raw_token.line, raw_token.column)


def _insert_placeholder_ws(tokens: list[Token]) -> list[Token]:
    padded_tokens = []
    for token, suc in zip(tokens[:-1], tokens[1:], strict=True):
        padded_tokens.append(token)
        if not (isinstance(token, Whitespace) or isinstance(suc, Whitespace)):
            padded_tokens.append(Whitespace("", token.line, suc.column))
    padded_tokens.append(tokens[-1])
    last_token = padded_tokens[-1]
    if not isinstance(last_token, Whitespace):
        padded_tokens.append(
            Whitespace(
                "",
                last_token.line,
                last_token.column + len(last_token.de_tokenize()),
            )
        )
    return padded_tokens


def _remove_eof(tokens: list[Token]) -> list[Token]:
    if tokens[-1].text == "<EOF>":
        return tokens[:-1]
    return tokens


def tokenize_java_code(code: str) -> list[Token]:
    """
    Tokenize a given code snippet into ProcessedTokens.
    :param code: The given code snippet.
    :return: Returns the ProcessedTokens
    """
    return _insert_placeholder_ws(
        _remove_eof(stream(lex_java(code)).map(_process_raw_token).to_list())
    )


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
    reports: frozenset[CheckstyleReport],
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
