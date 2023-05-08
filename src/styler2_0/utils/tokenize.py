import os
import re
from dataclasses import dataclass
from pathlib import Path

from antlr4 import CommonTokenStream, InputStream

from src.antlr.JavaLexer import JavaLexer


class ProcessedToken:
    def __init__(self, text: str, line: int, column: int) -> None:
        self.text = text
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return self.text

    def de_tokenize(self) -> str:
        return self.text


class Identifier(ProcessedToken):
    def __init__(self, text: str, line: int, column: int) -> None:
        super().__init__(text, line, column)

    def __str__(self) -> str:
        return "IDENTIFIER"


class Whitespace(ProcessedToken):
    def __init__(self, text: str, line: int, column: int) -> None:
        super().__init__(text, line, column)

    def __str__(self) -> str:
        return "_".join(
            self._process_whitespace_type(group)
            for group in re.findall(" +|\n+|\t+", self.text)
        )

    @staticmethod
    def _process_whitespace_type(whitespace: str) -> str:
        match list(whitespace):
            case ["\n", *_]:
                return f"{len(whitespace)}_NL"
            case ["\t", *_]:
                return f"{len(whitespace)}_TB"
            case [" ", *_]:
                return f"{len(whitespace)}_SP"
            case _:
                return "0_None"


class ProcessedSourceFile:
    def __init__(self, file_name: Path, tokens: list[ProcessedToken]) -> None:
        self.file_name = file_name
        self.tokens = tokens
        self._insert_placeholder_ws()

    def de_tokenize(self) -> str:
        return "".join(map(lambda token: token.de_tokenize(), self.tokens))

    def tokenized_str(self) -> str:
        return f"{' '.join(map(lambda token: str(token), self.tokens))}\n"

    def __str__(self) -> str:
        return self.tokenized_str()

    def _insert_placeholder_ws(self) -> None:
        padded_tokens = []
        for token, suc in zip(self.tokens[:-1], self.tokens[1:]):
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


class ContainsStr(str):
    def __eq__(self, other):
        return self.__contains__(other)


@dataclass
class RawToken:
    symbolic_name: str
    text: str
    line: int
    column: int

    def process(self) -> ProcessedToken:
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
    processed_java_files = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".java"):
                continue
            with open(Path(subdir) / Path(file)) as file_stream:
                content = file_stream.read()
                tokens = _tokenize_java_code(content)
                processed_java_file = ProcessedSourceFile(
                    Path(subdir) / Path(file), tokens
                )
                processed_java_files.append(processed_java_file)
    return processed_java_files
