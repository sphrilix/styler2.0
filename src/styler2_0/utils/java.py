from collections.abc import Callable
from dataclasses import dataclass

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener
from antlr4.Token import CommonToken
from streamerate import stream

from src.antlr.JavaLexer import JavaLexer
from src.antlr.JavaParser import JavaParser


@dataclass(eq=True, frozen=True)
class Lexeme:
    """
    Raw token passed from the lexer.
    """

    symbolic_name: str
    text: str
    line: int
    column: int


class NonParseableException(Exception):
    """
    Non parsable Exception.
    """


# As this inheriting from ErrorListener of ANTLR.
# noinspection PyPep8Naming
class SyntaxErrorListener(ErrorListener):
    """
    Syntax-Error listener used to spot errors while parsing.
    """

    def syntaxError(
        self,
        recognizer: JavaParser,
        offendingSymbol: CommonToken,
        line: int,
        column: int,
        msg: str,
        e: Exception,
    ) -> None:
        raise NonParseableException()


def returns_valid_java(func: Callable[..., ...]) -> Callable[..., ...]:
    """
    Decorator ensuring a decorated  function returns strings those strings are
    actually parseable.
    :param func: The decorated function.
    :return: Returns the decorated function.
    """

    def inner(*args: ..., **kwargs: ...) -> ...:
        return_values = func(*args, **kwargs)
        for return_value in return_values:
            if type(return_value) == str and not is_parseable(return_value):
                raise NonParseableException(f"Not valid java code: {return_value}")
        return return_values

    return inner


def is_parseable(code: str) -> bool:
    """
    Checks whether a given code snippet is parseable.
    :param code: The given code.
    :return: Returns True if parseable else False.
    """
    try:
        input_stream = InputStream(code)
        lexer = JavaLexer(input_stream)
        lexer.removeErrorListeners()
        lexer.addErrorListener(SyntaxErrorListener())
        token_stream = CommonTokenStream(lexer)
        parser = JavaParser(token_stream)
        parser.removeErrorListeners()
        parser.addErrorListener(SyntaxErrorListener())
        parser.compilationUnit()
    except NonParseableException:
        return False
    return True


def lex_java(code: str) -> list[Lexeme]:
    """
    Lex the given code snippet and return a list of lexemes.
    :param code: The given code snippet.
    :return: Returns the lexemes.
    """
    input_stream = InputStream(code)
    lexer = JavaLexer(input_stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(SyntaxErrorListener())
    token_stream = CommonTokenStream(lexer)
    token_stream.fill()
    return (
        stream(token_stream.tokens)
        .map(
            lambda common: Lexeme(
                symbolic_name=lexer.symbolicNames[common.type],
                line=common.line,
                column=common.column,
                text=common.text,
            )
        )
        .to_list()
    )
