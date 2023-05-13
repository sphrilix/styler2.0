import random
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from pathlib import Path

from streamerate import stream

from src.styler2_0.utils.java import NonParseableException, returns_valid_java
from src.styler2_0.utils.tokenize import Whitespace, tokenize_java_code
from styler2_0.utils.utils import retry


def _insert(char: str, string: str) -> str:
    return string + char


def _delete(char: str, string: str) -> str:
    return string.replace(char, "", 1)


class Operation(Enum):
    """
    Enum of operations that can be performed.
    """

    INSERT_SPACE = partial(_insert, " ")
    INSERT_TAB = partial(_insert, "\t")
    INSERT_NL = partial(_insert, "\n")
    DELETE_SPACE = partial(_delete, " ")
    DELETE_NL = partial(_delete, "\n")

    def __call__(self, *args: str, **kwargs: ...) -> str:
        return self.value(args[0])


def _pick_random_operation() -> Operation:
    return random.choice(list(Operation))


class ViolationGenerator(ABC):
    """
    Abstract violation generator base class.
    All generators should inherit from these base class.
    """

    def __init__(
        self, non_violated_source: str, checkstyle_config: Path, checkstyle_version: str
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
        operation = _pick_random_operation()
        tokens = tokenize_java_code(self.non_violated_source)
        random_token = random.choice(
            stream(tokens).filter(lambda token: isinstance(token, Whitespace)).to_list()
        )
        random_token.text = operation(random_token.text)

        return self.non_violated_source, "".join(
            stream(tokens).map(lambda token: token.de_tokenize())
        )
