import copy
import os
import re
from collections.abc import Generator
from contextlib import suppress
from pathlib import Path
from typing import Self

from streamerate import stream

from src.styler2_0.utils.checkstyle import (
    CheckstyleFileReport,
    Violation,
    ViolationType,
)
from src.styler2_0.utils.java import Lexeme, lex_java

#######################################################################################
# DISCLAIMER!                                                                         #
# The preprocessing is not entirely identical because in the Styler preprocessing, 2  #
# bugs were identified:                                                               #
#                                                                                     #
# 1. Whitespaces before line breaks were simply omitted.                              #
#                                                                                     #
# 2. If a whitespace contained tabs and spaces, it was encoded only with the first    #
#   type and the second one was simply omitted.                                       #
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
            return violation.column <= end_col
        return False


class LiteralToken(Token):
    def __init__(self, text: str, line: int, column: int, literal_type: str) -> None:
        super().__init__(text, line, column)
        self.literal_type = literal_type

    def __str__(self) -> str:
        return self.literal_type


class Identifier(Token):
    """
    Class which represents Identifier/Literals
    """

    STRING_SPLITTER = re.compile(
        r"(?<=[a-z])(?=[A-Z])|(_)|(-)|(\\d)|(?<=[A-Z])(?=[A-Z][a-z])|\\s+"
    )
    ALL_LOWER_SUB_TOKEN = "[I_LOWER]"
    ALL_UPPER_SUB_TOKEN = "[I_UPPER]"
    FIRST_UPPER_OTHER_LOWER_SUB_TOKEN = "[I_FIRST_UPPER_OTHER_LOWER]"
    HYPHEN = "[I_HYPHEN]"
    UNDERSCORE = "[I_UNDERSCORE]"

    def __str__(self) -> str:
        sub_tokens = self.sub_tokens_of_identifier(self.text)
        return " ".join(map(self._process_sub_token, sub_tokens))

    @staticmethod
    def sub_tokens_of_identifier(identifier: str) -> list[str]:
        return list(
            filter(
                lambda token: token is not None and token != "",
                Identifier.STRING_SPLITTER.split(identifier),
            )
        )

    def _process_sub_token(self, sub_token: str) -> str:
        if sub_token.isupper():
            return self.ALL_UPPER_SUB_TOKEN
        if sub_token[0].isupper() and sub_token[1:].islower():
            return self.FIRST_UPPER_OTHER_LOWER_SUB_TOKEN
        if sub_token == "-":
            return self.HYPHEN
        if sub_token == "_":
            return self.UNDERSCORE
        return self.ALL_LOWER_SUB_TOKEN

    @staticmethod
    def parse_tokenized_str(tokenized_str: list[str], old_name: str) -> str:
        """
        Parse a tokenized string to the real representation.
        :param tokenized_str: The tokenized string representation.
        :param old_name: The old name of the identifier in the wrong format.
        :return: Returns the new name of the identifier in the correct format.
        """

        # Filter non name format tokens
        tokenized_str = [t for t in tokenized_str if t.startswith("[I")]
        out = ""
        tokens = [t for t in Identifier.sub_tokens_of_identifier(old_name) if t != "_"]
        for current_template in tokenized_str:
            if current_template == Identifier.UNDERSCORE:
                out += "_"
                continue
            if not tokens:
                break
            t = tokens.pop(0)
            if current_template == Identifier.ALL_LOWER_SUB_TOKEN:
                out += t.lower()
            elif current_template == Identifier.ALL_UPPER_SUB_TOKEN:
                out += t.upper()
            elif current_template == Identifier.FIRST_UPPER_OTHER_LOWER_SUB_TOKEN:
                out += t.title()
        if tokens:
            out += "".join(tokens)
        return out


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

    @staticmethod
    def parse_tokenized_str(tokenized_str: str) -> str:
        """
        Process a tokenized whitespace to the real representation.
        :param tokenized_str: The tokenized str representation.
        :return: Returns the real representation.
        """
        splits = re.sub(r"(ID|DD)_", "", tokenized_str).split("_")
        types = splits[1::2]
        amounts = splits[::2]
        sub_parts = []
        for ws_type, amount in zip(types, amounts, strict=True):
            match ws_type:
                case "TB":
                    sub_str = "\t"
                case "SP":
                    sub_str = " "
                case "NL":
                    sub_str = "\n"
                case "None":
                    sub_str = ""
                case _:
                    raise ValueError("Non whitespace str passed.")
            sub_parts.append(f"{int(amount)*sub_str}")
        return "".join(sub_parts)


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

    def de_tokenize(self) -> str:
        """
        Return none tokenized representation that in this case is the empty string,
        as checkstyle tokens are not part of the original source code.
        :return:
        """
        return ""


class ProcessedSourceFile:
    """
    Class which represents a processed/tokenized Java file.
    """

    def __init__(
        self,
        file_name: Path | None,
        tokens: list[Token],
        report: CheckstyleFileReport = None,
    ) -> None:
        self.file_name = file_name
        self.tokens = tokens

        self.non_ws_tokens = (
            stream(self.tokens)
            .filter(lambda token: not isinstance(token, Whitespace))
            .to_list()
        )

        self.report = report
        self.checkstyle_tokens = []

        # TODO: check if this is really needed, I don't think so
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

    def tokens_between_violations(self) -> Generator[list[list[Token]], None, None]:
        """
        Get all tokens between two violation tags.
        :return: Returns a generator which yields all tokens between two violation tags.
        """
        violation_tag_pairs = zip(
            self.checkstyle_tokens[::2], self.checkstyle_tokens[1::2], strict=True
        )
        for violation_tag_pair in violation_tag_pairs:
            start, end = violation_tag_pair
            yield self.tokens[self.tokens.index(start) : self.tokens.index(end) + 1]

    def violations_with_ctx(
        self, context: int = 2
    ) -> Generator[list[Token], None, None]:
        """
        Get all tokens within a specified around a violation.
        :param context: The specified line context.
        :return: Returns the tokens within the specified context around a violation.
        """
        for violated_tokens in self.tokens_between_violations():
            ctx_start_line = max(1, violated_tokens[0].line - context)
            ctx_end_line = min(self.tokens[-1].line, violated_tokens[-1].line + context)
            yield list(
                stream(self.tokens).filter(
                    lambda t, s=ctx_start_line, e=ctx_end_line: s <= t.line <= e
                )
            )

    def get_fixes_for(
        self, fixes: list[list[str]], violation: (CheckstyleToken, CheckstyleToken)
    ) -> Generator[Self, None, None]:
        """
        Get all possible fixes for a violation.
        Replace the tokens between the violation tags with the fixes.
        If fixes longer than tokens between tokens, just insert the first n tokens.
        If vice versa just fill with old tokens.
        Like styler does.
        :param fixes: The possible fixes.
        :param violation: The violation to be tackled.
        :return: Returns the processed source file with the fixes applied.
        """
        start, end = violation
        start_idx = self.tokens.index(start)
        end_idx = self.tokens.index(end)
        tokens_between = self.tokens[start_idx : end_idx + 1]
        possible_fixes = []
        for fix in fixes:
            # TODO: why ValueError?
            with suppress(ValueError):
                if start.text.lower().endswith("name"):
                    identifier = next(
                        iter(t for t in tokens_between if isinstance(t, Identifier))
                    )
                    possible_fix = self._insert_fix_name_violation(fix, identifier)
                else:
                    possible_fix = self._insert_fix_format_violation(
                        fix, tokens_between
                    )
                possible_fixes.append(possible_fix)
        for possible_fix in possible_fixes:
            copy_tokens = copy.deepcopy(self.tokens)
            if start.text.lower().endswith("name"):
                identifier = next(
                    iter(t for t in tokens_between if isinstance(t, Identifier))
                )
                old_value = identifier.text
                new_value = possible_fix[0].text
                for t in copy_tokens:
                    if isinstance(t, Identifier) and t.text == old_value:
                        t.text = new_value
            else:
                copy_tokens[start_idx : end_idx + 1] = possible_fix
            yield ProcessedSourceFile(self.file_name, copy_tokens)

    def __repr__(self) -> str:
        return self.tokenized_str()

    def _insert_checkstyle_report(self, report: CheckstyleFileReport) -> None:
        assert (
            report.path == self.file_name or self.file_name is None
        ), "Report and source file path must match."

        for violation in report.violations:
            if violation.type.name.endswith("NAME"):
                start, end = self._get_name_violation_ctx(violation)
            elif violation.column is None:
                start, end = self._get_line_violation_ctx(violation)
            else:
                start, end = self._get_col_violation_ctx(violation)

            start_style_token = CheckstyleToken(
                violation.type.value, start.line, start.column, True
            )
            end_style_token = CheckstyleToken(
                violation.type.value, end.line, end.column, False
            )

            self.checkstyle_tokens.append(start_style_token)
            self.checkstyle_tokens.append(end_style_token)

            start_idx = max(0, self.tokens.index(start))
            end_idx = min(len(self.tokens), self.tokens.index(end) + 1)
            self.tokens.insert(end_idx, end_style_token)
            self.tokens.insert(start_idx, start_style_token)

    def _get_line_violation_ctx(self, violation: Violation) -> (Token, Token):
        assert violation.column is None
        if violation.line <= 1:
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
                self.tokens[0],
            )
        return next(
            (
                non_ws_token
                for non_ws_token in self.non_ws_tokens
                if non_ws_token.line == token.line
                and token.column < non_ws_token.column
                or token.line < non_ws_token.line
            ),
            self.tokens[-1],
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

    def _get_name_violation_ctx(self, violation: Violation) -> tuple[Token, Token]:
        assert violation.type.name.endswith("NAME")
        assert violation.column is not None
        violated_name_token = (
            stream(self.non_ws_tokens)
            .reversed()
            .filter(lambda token: isinstance(token, Identifier))
            .filter(lambda token: token.line == violation.line)
            .filter(lambda token: token.column <= violation.column)
            .next()
        )
        return violated_name_token, violated_name_token

    @staticmethod
    def _insert_fix_format_violation(
        fix: list[str], tokens_between: list[Token]
    ) -> list[Token]:
        fix = [f for f in fix if not f.startswith("[I") and not f.startswith("<")]
        possible_fix = []
        for token in tokens_between:
            if isinstance(token, Whitespace) and len(fix) > 0:
                fix_str = Whitespace.parse_tokenized_str(fix[0])
                fix_token = copy.deepcopy(token)
                fix_token.text = fix_str
                possible_fix.append(fix_token)
                fix = fix[1:]
            else:
                possible_fix.append(token)
        return possible_fix

    @staticmethod
    def _insert_fix_name_violation(
        fix: list[str], identifier: Identifier
    ) -> list[Token]:
        assert isinstance(identifier, Identifier)
        new_value = Identifier.parse_tokenized_str(fix, identifier.text)
        return [Identifier(new_value, identifier.line, identifier.column)]


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
        case "IDENTIFIER":
            return Identifier(raw_token.text, raw_token.line, raw_token.column)
        case "WS":
            return Whitespace(raw_token.text, raw_token.line, raw_token.column)
        case "COMMENT":
            return Comment(raw_token.text, raw_token.line, raw_token.column)
        case "LITERAL":
            return LiteralToken(
                raw_token.text,
                raw_token.line,
                raw_token.column,
                raw_token.symbolic_name,
            )
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


# TODO: Use utility function to read files
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


def interesting_tokens_type(violation: ViolationType) -> type(Token):
    """
    Get the interesting token type for a given violation.
    :param violation: The given violation.
    :return: Returns the interesting token type.
    """
    match violation.name.split("_")[-1]:
        case "NAME":
            return Identifier
        case _:
            return Whitespace
