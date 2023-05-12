from collections.abc import Callable

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener
from antlr4.Token import CommonToken

from src.antlr.JavaLexer import JavaLexer
from src.antlr.JavaParser import JavaParser


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


def assure_valid_java_code_return(func: Callable[..., ...]) -> Callable[..., ...]:
    """
    Decorator ensuring a decorated  function returns parseable java code.
    :param func: The decorated function.
    :return: Returns the decorated function.
    """

    def inner(*args: ..., **kwargs: ...) -> ...:
        return_values = func(*args, **kwargs)
        for return_value in return_values:
            if not is_parseable(return_value):
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
