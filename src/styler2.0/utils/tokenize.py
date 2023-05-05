import re
from dataclasses import dataclass
from typing import List

from antlr4 import CommonTokenStream, InputStream

from src.antlr.JavaLexer import JavaLexer


class ProcessedToken:
    def __init__(self, text: str, line: int, column: int) -> None:
        self.text = text
        self.line = line
        self.column = column

    def __str__(self) -> str:
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
            self._process_whitespace_type(group) for group in re.findall(" +|\n+|\t+", self.text)
        )

    @staticmethod
    def _process_whitespace_type(whitespace: str) -> str:
        match [*whitespace]:
            case ["\n", *_]:
                return f"{len(whitespace)}_NL"
            case ["\t", *_]:
                return f"{len(whitespace)}_TB"
            case [" ", *_]:
                return f"{len(whitespace)}_SP"
            case _:
                return "0_None"


@dataclass
class RawToken:
    symbolic_name: str
    text: str
    line: int
    column: int

    def process(self) -> ProcessedToken:
        match self.symbolic_name:
            case "IDENTIFIER":
                return Identifier(self.text, self.line, self.column)
            case "WS":
                return Whitespace(self.text, self.line, self.column)
            case _:
                return ProcessedToken(self.text, self.line, self.column)


def lex_java(code: str) -> List[ProcessedToken]:
    input_stream = InputStream(code)
    lexer = JavaLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    token_stream.fill()
    return [RawToken(lexer.symbolicNames[token.type], token.text, token.line, token.column).process() for token in
            token_stream.tokens]


class ProcessedSourceFile:
    def __init__(self, content: str, processed_content: str) -> None:
        self.content = content
        self.processed_content = processed_content
