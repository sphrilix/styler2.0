import json
import os
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Self

from streamerate import stream
from tqdm import tqdm

from src.styler2_0.utils.checkstyle import (
    WrongViolationAmountException,
    returns_n_violations,
    run_checkstyle_on_dir,
)
from src.styler2_0.utils.java import NonParseableException, returns_valid_java
from src.styler2_0.utils.tokenize import (
    CheckstyleToken,
    ProcessedSourceFile,
    Token,
    Whitespace,
    tokenize_java_code,
)
from src.styler2_0.utils.utils import (
    TooManyTriesException,
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
        self._apply_to_token(token)

    @abstractmethod
    def _apply(self, code: str) -> str:
        pass

    def _apply_to_token(self, token: Token) -> None:
        token.text = self._apply(token.text)

    def is_applicable(self, token: Token, context: tuple[Token, Token]) -> bool:
        """
        Determines if the operator is applicable to the given token.
        :param context:
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

    def is_applicable(self, token: Token, context: tuple[Token, Token]) -> bool:
        return isinstance(token, Whitespace)


class DeleteOperation(CharOperation):
    """
    Deletion of a character operation.
    """

    def _apply(self, code: str) -> str:
        return _delete(self.char, code)

    def is_applicable(self, token: Token, context: tuple[Token, Token]) -> bool:
        return (
            super().is_applicable(token, context)
            and self.char in token.text
            and (len(token.text) > 1 or self._suitable_del_ctx(context))
        )

    @staticmethod
    def _suitable_del_ctx(context: tuple[Token, Token]) -> bool:
        return (
            context[0].is_operator()
            or context[0].is_punctuation()
            or context[1].is_operator()
            or context[1].is_punctuation()
        )


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


class ViolationGenerator(ABC):
    """
    Abstract class for generating violations.
    Inheriting classes must implement the pre- and post-processing steps of generating
    violations. Besides, that the implementation of generating a violation must be
    done.
    """

    def __init__(
        self,
        n: int,
        non_violated_sources_dir: Path,
        checkstyle_config: Path,
        checkstyle_version: str,
        save_path: Path,
        delta: int = 3 * 60 * 60,
    ) -> None:
        self.n = n
        self.non_violated_sources = get_files_in_dir(non_violated_sources_dir, ".java")
        self.checkstyle_config = checkstyle_config
        self.checkstyle_version = checkstyle_version
        self.save_path = save_path
        self.delta = delta

    @abstractmethod
    def _preprocessing_steps(self) -> None:
        """
        Implement this if violation generation needs any pre-processing steps.
        :return:
        """

    @abstractmethod
    def _postprocessing_steps(self) -> None:
        """
        Implement this if violation generation needs any post-processing steps
        :return:
        """

    @retry(
        n=3,
        exceptions=(
            NonParseableException,
            WrongViolationAmountException,
            OperationNonApplicableException,
        ),
    )
    @returns_n_violations(
        n=1,
        use_instance=True,
        checkstyle_version="checkstyle_version",
        checkstyle_config="checkstyle_config",
    )
    @returns_valid_java
    def __generate_violation(self, tokens: list[Token]) -> str:
        """
        This is the actual method called to generate the violation.
        It ensures that the code is parseable and returns exactly 1 violation.
        Also, the generation is tried 3 times if one of the above regulations
        is violated.
        :param tokens: Tokens of the non violated file.
        :return: Returns violated Java code.
        """
        return self._generate_violation(tokens)

    @abstractmethod
    def _generate_violation(self, tokens: list[Token]) -> str:
        """
        Implementation of the violation generation.
        Must be implemented by inheriting classes.
        :param tokens:
        :return: Returns
        """

    def generate_violations(self) -> None:
        """

        :return:
        """
        self._preprocessing_steps()
        os.makedirs(self.save_path, exist_ok=True)
        with tqdm(total=self.n) as progress_bar:
            start = time.time()
            valid_violations = 0
            while valid_violations < self.n and time.time() - start < self.delta:
                with suppress(TooManyTriesException):
                    current_file = random.choice(self.non_violated_sources)
                    content = read_content_of_file(current_file)
                    tokens = tokenize_java_code(content)
                    non_violated = "".join(
                        stream(tokens).map(lambda token: token.de_tokenize())
                    )
                    violated = self.__generate_violation(tokens)
                    current_save_path = self.save_path / Path(str(valid_violations))
                    os.makedirs(current_save_path, exist_ok=True)
                    non_violated_file_name = Path(current_file.name)
                    violated_file_name = Path(f"VIOLATED_{current_file.name}")
                    save_content_to_file(
                        current_save_path / non_violated_file_name, non_violated
                    )
                    save_content_to_file(
                        current_save_path / violated_file_name, violated
                    )
                    valid_violations += 1
                    progress_bar.update()
        self._postprocessing_steps()
        self._generate_metadata()

    def _generate_metadata(self) -> None:
        for sample_dir in os.listdir(self.save_path):
            curr_dir = self.save_path / Path(sample_dir)
            for _, _, files in os.walk(curr_dir):
                violated, non_violated = None, None
                for file in files:
                    if file.startswith("VIOLATED"):
                        violated = curr_dir / Path(file)
                    elif file.endswith(".java"):
                        non_violated = curr_dir / Path(file)
                if not non_violated or not violated:
                    continue
                metadata = Metadata(
                    non_violated,
                    violated,
                    self.checkstyle_config,
                    self.checkstyle_version,
                )
                metadata.save_to_directory(curr_dir)


class RandomGenerator(ViolationGenerator):
    """
    Random violations generator.
    """

    def _preprocessing_steps(self) -> None:
        pass

    def _postprocessing_steps(self) -> None:
        pass

    def _generate_violation(self, tokens: list[Token]) -> str:
        operation = random.choice(list(Operations)).value
        print(operation)
        applicable_tokens = self._get_applicable_tokens(operation, tokens)
        if not applicable_tokens:
            raise OperationNonApplicableException(
                f"{operation} not applicable on token sequence."
            )
        random_token = random.choice(applicable_tokens)
        operation(random_token)
        return "".join(stream(tokens).map(lambda token: token.de_tokenize()))

    @staticmethod
    def _get_applicable_tokens(
        operation: Operation, tokens: list[Token]
    ) -> list[Token]:
        padded_tokens: list[Token] = (
            [Whitespace("", 0, 0)] + tokens + [Whitespace("", 0, 0)]
        )
        return list(
            stream(
                zip(
                    padded_tokens[1:-1],
                    zip(padded_tokens[:-2], padded_tokens[2:], strict=True),
                    strict=True,
                )
            )
            .filter(
                lambda token_with_ctx: operation.is_applicable(
                    token_with_ctx[0], token_with_ctx[1]
                )
            )
            .map(lambda token_with_ctx: token_with_ctx[0])
            .to_list()
        )


@dataclass(frozen=True)
class ThreeGram:
    token_before: Token | None
    whitespace: Whitespace
    token_after: Token | None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ThreeGram):
            return False
        return (
            str(self.token_before) == str(other.token_before)
            and str(self.whitespace) == str(other.whitespace)
            and str(self.token_after) == str(other.token_after)
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __repr__(self) -> str:
        return (
            f"[{str(self.token_before)}, "
            f"{str(self.whitespace)}, "
            f"{str(self.token_after)}]"
        )


class ThreeGramGenerator(ViolationGenerator):
    """
    3-gram violation generator.
    """

    def _preprocessing_steps(self) -> None:
        self.__collect_3_grams()

    def _postprocessing_steps(self) -> None:
        pass

    def _generate_violation(self, tokens: list[Token]) -> str:
        three_gram = random.choice(list(self.__build_3_grams(tokens)))
        alternatives_with_prob = self.__get_alternatives_with_prob(three_gram)
        if not alternatives_with_prob:
            raise OperationNonApplicableException
        weights = list(alternatives_with_prob.values())
        choices = list(
            stream(list(alternatives_with_prob.keys())).map(
                lambda t_gram: t_gram.whitespace
            )
        )
        new_ws = random.choices(choices, weights)
        three_gram.whitespace.text = new_ws[0].text
        return "".join(stream(tokens).map(lambda token: token.de_tokenize()))

    def __collect_3_grams(self) -> None:
        self.three_grams = {}
        for source in self.non_violated_sources:
            tokens = tokenize_java_code(read_content_of_file(source))
            three_grams = self.__build_3_grams(tokens)
            for three_gram in three_grams:
                self.three_grams.setdefault(three_gram, 0)
                self.three_grams[three_gram] += 1

    @staticmethod
    def __build_3_grams(tokens: list[Token]) -> Generator[ThreeGram, Any, None]:
        whitespaces = (
            stream(tokens).filter(lambda token: isinstance(token, Whitespace)).to_list()
        )
        for whitespace in whitespaces:
            idx = tokens.index(whitespace)
            token_before = None if idx == 0 else tokens[idx - 1]
            token_after = None if idx == len(tokens) - 1 else tokens[idx + 1]
            yield ThreeGram(token_before, whitespace, token_after)

    def __get_alternatives_with_prob(
        self, three_gram: ThreeGram
    ) -> dict[ThreeGram, float]:
        alternatives = (
            stream(list(self.three_grams.items()))
            .filter(
                lambda entry: str(entry[0].token_before) == str(three_gram.token_before)
                and str(entry[0].token_after) == str(three_gram.token_after)
                and str(entry[0].whitespace) != str(three_gram.whitespace)
            )
            .to_dict()
        )
        total = (
            alternatives.items()
            .map(lambda entry: entry[1])
            .reduce(lambda entry1, entry2: entry1 + entry2, 0)
        )
        return dict(
            alternatives.items()
            .map(lambda entry: (entry[0], entry[1] / total))
            .to_dict()
        )


class Protocol(Enum):
    RANDOM = RandomGenerator
    THREE_GRAM = ThreeGramGenerator

    def __call__(self, *args, **kwargs) -> ViolationGenerator:
        return self.value(*args, **kwargs)


class Metadata:
    def __init__(
        self,
        non_violated_source: Path,
        violated_source: Path,
        config: Path,
        version: str,
        non_violated_str: str | None = None,
        violated_str: str | None = None,
    ) -> None:
        self.non_violated_source = non_violated_source
        self.violated_source = violated_source
        self.config = config
        self.version = version
        if not non_violated_str or not violated_str:
            self.__set_up_metadata()
        else:
            self.violated_str = violated_str
            self.non_violated_str = non_violated_str

    def to_json(self) -> str:
        json_dict = {
            "non_violated_source": str(self.non_violated_source),
            "violated_source": str(self.violated_source),
            "config": str(self.config),
            "version": self.version,
            "non_violated_str": self.non_violated_str,
            "violated_Str": self.violated_str,
        }
        return json.dumps(json_dict)

    @classmethod
    def from_json(cls, json_data: str) -> Self:
        data = json.loads(json_data)
        return cls(
            Path(data["non_violated_source"]),
            Path(data["violated_source"]),
            Path(data["config"]),
            data["version"],
            data["non_violated_str"],
            data["violated_str"],
        )

    def save_to_directory(self, directory: Path) -> None:
        filepath = directory / Path("data.json")
        save_content_to_file(filepath, self.to_json())

    def __set_up_metadata(self) -> None:
        reports = run_checkstyle_on_dir(
            self.non_violated_source.parents[0], self.version, self.config
        )
        processed_files = [
            ProcessedSourceFile(
                report.path,
                tokenize_java_code(read_content_of_file(report.path)),
                report,
            )
            for report in reports
            if str(report.path).endswith(".java")
        ]
        non_violated = (
            stream(processed_files)
            .filter(lambda file: len(file.report.violations) == 0)
            .next()
        )
        violated = (
            stream(processed_files)
            .filter(lambda file: len(file.report.violations) > 0)
            .next()
        )
        self.non_violated_str, self.violated_str = self.__filter_relevant_tokens(
            non_violated, violated
        )

    @staticmethod
    def __filter_relevant_tokens(
        non_violated: ProcessedSourceFile, violated: ProcessedSourceFile
    ) -> (str, str):
        assert len(violated.checkstyle_tokens) == 2
        start_check, end_check = violated.checkstyle_tokens
        violated_tokens = violated.tokens[
            violated.tokens.index(start_check) : violated.tokens.index(end_check) + 1
        ]
        violated_tokens_wo_checkstyle = (
            stream(violated.tokens)
            .filter(lambda token: not isinstance(token, CheckstyleToken))
            .to_list()
        )
        start_idx_non_violated = violated_tokens_wo_checkstyle.index(violated_tokens[1])
        end_idx_non_violated = (
            violated_tokens_wo_checkstyle.index(violated_tokens[-2]) + 1
        )
        non_violated_tokens = non_violated.tokens[
            start_idx_non_violated:end_idx_non_violated
        ]
        return (
            " ".join(stream(non_violated_tokens).map(str)),
            " ".join(stream(violated_tokens).map(str)),
        )


def generate_n_violations(
    n: int,
    protocol: Protocol,
    non_violated_sources: Path,
    checkstyle_config: Path,
    checkstyle_version: str,
    save_path: Path,
    delta: int = 3 * 60 * 60,  # 3h
) -> None:
    """
    Create n violation out of non_violated_sources using the provided protocol.
    :param n: The number of violations.
    :param protocol: The protocol used to generate.
    :param non_violated_sources: The sources out of which the violations should be
                                 generated.
    :param checkstyle_config: Checkstyle config.
    :param checkstyle_version: Version of checkstyle.
    :param save_path: Where to save the violations.
    :param delta: Max time this generation is allowed to run.
    :return:
    """
    save_path = save_path / Path(protocol.name.lower())
    protocol(
        n, non_violated_sources, checkstyle_config, checkstyle_version, save_path, delta
    ).generate_violations()
