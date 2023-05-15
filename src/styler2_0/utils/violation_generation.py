import random
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from streamerate import stream

from src.styler2_0.utils.java import NonParseableException, returns_valid_java
from src.styler2_0.utils.tokenize import Token, Whitespace
from src.styler2_0.utils.utils import retry


def _insert(char: str, string: str) -> str:
    return string + char


def _delete(char: str, string: str) -> str:
    return string.replace(char, "", 1)


class OperationNonApplicableException(Exception):
    """
    Exception that is raised whenever an operator is not applicable to the
    given code snippet.
    """


class Operation(ABC):
    """
    Operation applicable to code.
    """

    def __call__(self, token: Token) -> None:
        if not self.is_applicable(token):
            raise OperationNonApplicableException("Operation not applicable on tokens.")
        self._apply_to_token(token)

    @abstractmethod
    def _apply(self, code: str) -> str:
        pass

    def _apply_to_token(self, token: Token) -> None:
        token.text = self._apply(token.text)

    def is_applicable(self, token: Token) -> bool:
        """
        Determines if the operator is applicable to the given token.
        :param token: Token to be checked.
        :return: Returns True if applicable else False.
        """
        return True


class CharOperation(Operation, ABC):
    """
    Operation altering a single char.
    """

    def __init__(self, char: str) -> None:
        self.char = char

    def is_applicable(self, token: Token) -> bool:
        return isinstance(token, Whitespace)


class DeleteOperation(CharOperation):
    """
    Deletion of a character operation.
    """

    def _apply(self, code: str) -> str:
        return _delete(self.char, code)

    def is_applicable(self, token: Token) -> bool:
        return super().is_applicable(token) and self.char in token.text


class InsertOperation(CharOperation):
    """
    Insertion of a character operator.
    """

    def _apply(self, code: str) -> str:
        return _insert(self.char, code)


class Operations(Enum):
    """
    Enum of operations that can be performed.
    """

    INSERT_SPACE = InsertOperation(" ")
    INSERT_TAB = InsertOperation("\t")
    INSERT_NL = InsertOperation("\n")
    DELETE_TAB = DeleteOperation("\t")
    DELETE_SPACE = DeleteOperation(" ")
    DELETE_NL = DeleteOperation("\n")

    def __call__(self, code: str) -> str:
        return self.value(code)


def _pick_random_operation() -> Operations:
    return random.choice(list(Operations))


class ViolationGenerator(ABC):
    """
    Abstract violation generator base class.
    All generators should inherit from these base class.
    """

    def __init__(
        self,
        non_violated_source: list[Token],
        checkstyle_config: Path,
        checkstyle_version: str,
    ) -> None:
        self.non_violated_source = non_violated_source
        self.checkstyle_config = checkstyle_config
        self.checkstyle_version = checkstyle_version

    @retry(n=10, exceptions=NonParseableException)
    @returns_valid_java
    def generate_violation(self) -> (str, str):
        """
        Generate parseable code with exactly one violation in it.
        :return: Returns original code, altered code with exactly one exception
        """
        return self._generate_violation()

    @abstractmethod
    def _generate_violation(self) -> (str, str):
        pass


class RandomGenerator(ViolationGenerator):
    """
    Generator for generating violations randomly.
    """

    def _generate_violation(self) -> (str, str):
        operation = _pick_random_operation().value()
        random_token = random.choice(
            stream(self.non_violated_source)
            .filter(lambda token: operation.is_applicable(token))
            .to_list()
        )
        operation(random_token)
        return self.non_violated_source, "".join(
            stream(self.non_violated_source).map(str)
        )
