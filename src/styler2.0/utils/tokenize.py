import re
from dataclasses import dataclass
from typing import List
from antlr4 import InputStream, CommonTokenStream

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
        super(Identifier, self).__init__(text, line, column)

    def __str__(self) -> str:
        return "IDENTIFIER"


class Whitespace(ProcessedToken):
    def __init__(self, text: str, line: int, column: int) -> None:
        super(Whitespace, self).__init__(text, line, column)

    def __str__(self) -> str:
        return ""

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
