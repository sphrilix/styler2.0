import os
import random
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from streamerate import stream
from tqdm import tqdm

from src.styler2_0.utils.java import NonParseableException, returns_valid_java
from src.styler2_0.utils.tokenize import Token, Whitespace, tokenize_java_code
from src.styler2_0.utils.utils import (
    get_files_in_dir,
    read_content_of_file,
    retry,
    save_content_to_file,
)


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
        tokens: list[Token],
        checkstyle_config: Path,
        checkstyle_version: str,
    ) -> None:
        self.tokens = tokens
        self.checkstyle_config = checkstyle_config
        self.checkstyle_version = checkstyle_version

    @retry(n=10, exceptions=(NonParseableException, OperationNonApplicableException))
    @returns_valid_java
    def generate_violation(self) -> (str, str):
        """
        Generate parseable code with exactly one violation in it.
        :return: Returns original code, altered code with exactly one exception
        """

        non_violated = "".join(
            stream(self.tokens).map(lambda token: token.de_tokenize())
        )
        return non_violated, self._generate_violation()

    @abstractmethod
    def _generate_violation(self) -> str:
        pass


class RandomGenerator(ViolationGenerator):
    """
    Generator for generating violations randomly.
    """

    def _generate_violation(self) -> str:
        operation = _pick_random_operation().value
        applicable_tokens = (
            stream(self.tokens)
            .filter(lambda token: operation.is_applicable(token))
            .to_list()
        )
        if not applicable_tokens:
            raise OperationNonApplicableException(
                f"{operation} not applicable on token sequence."
            )
        random_token = random.choice(applicable_tokens)
        operation(random_token)
        return "".join(stream(self.tokens).map(lambda token: token.de_tokenize()))


class Protocol(Enum):
    RANDOM = "RANDOM"

    def get_generator(
        self, non_violated_source: list[Token], checkstyle_config: Path, version: str
    ) -> ViolationGenerator:
        match self:
            case Protocol.RANDOM:
                return RandomGenerator(non_violated_source, checkstyle_config, version)

    def __str__(self) -> str:
        return self.value


def generate_n_violations(
    n: int,
    protocol: Protocol,
    non_violated_sources: Path,
    checkstyle_config: Path,
    checkstyle_version: str,
    save_path: Path,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    for i in tqdm(range(n)):
        non_violated_source_files = get_files_in_dir(
            non_violated_sources, suffix=".java"
        )
        current_file = random.choice(non_violated_source_files)
        content = read_content_of_file(current_file)
        tokens = tokenize_java_code(content)
        generator = protocol.get_generator(
            tokens, checkstyle_config, checkstyle_version
        )
        non_violated, violated = generator.generate_violation()
        current_save_path = save_path / Path(str(i))
        os.makedirs(current_save_path, exist_ok=True)
        non_violated_file_name = Path(current_file.name)
        violated_file_name = Path(f"Violated{current_file.name}")
        save_content_to_file(current_save_path / non_violated_file_name, non_violated)
        save_content_to_file(current_save_path / violated_file_name, violated)
