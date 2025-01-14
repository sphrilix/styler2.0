import copy
import csv
import json
import os
import random
import shutil
import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Self
from xml.etree.ElementTree import ParseError

from streamerate import stream
from tqdm import tqdm

from src.styler2_0.utils.checkstyle import run_checkstyle_on_dir, run_checkstyle_on_strs
from src.styler2_0.utils.java import NonParseableException, returns_valid_java
from src.styler2_0.utils.tokenize import (
    CheckstyleToken,
    Identifier,
    ProcessedSourceFile,
    Token,
    Whitespace,
    interesting_tokens_type,
    tokenize_java_code,
)
from src.styler2_0.utils.utils import (
    TooManyTriesException,
    get_files_in_dir,
    read_content_of_file,
    retry,
    save_content_to_file,
)

CURR_DIR = os.path.dirname(os.path.relpath(__file__))
CSV_PATH = Path(os.path.join(CURR_DIR, "../../../csv/three_grams.csv"))
DEFAULT_GEN_BATCH_SIZE = 100


def _insert(char: str, string: str) -> str:
    if random.random() < 0.5:
        return char + string
    return string + char


def _delete(char: str, string: str) -> str:
    return string.replace(char, "")


class MetadataException(Exception):
    """
    Exception that is raised whenever the metadata cannot be calculated.
    """


class IdentifierAlteringOperations(Enum):
    """
    Enum of operations that can be performed on an identifier.
    """

    TO_SNAKE_CASE = partial(
        lambda name: "_".join(Identifier.sub_tokens_of_identifier(name)).lower()
    )
    TO_CONSTANT_CASE = partial(
        lambda name: "_".join(Identifier.sub_tokens_of_identifier(name)).upper()
    )
    TO_CAMEL_CASE = partial(
        lambda name: Identifier.sub_tokens_of_identifier(name)[0].lower()
        + "".join(Identifier.sub_tokens_of_identifier(name)[1:]).title()
    )

    def __call__(self, token: str) -> str:
        return self.value(token)


class TokenAlteringOperations(Enum):
    """
    Enum of operations that can be performed on a token.
    """

    TO_LOWER = partial(lambda name: name.lower())
    TO_UPPER = partial(lambda name: name.upper())
    TO_CAMEL_CASE = partial(lambda name: name.title())
    TO_SNAKE_CASE = partial(
        lambda name: f"_{name.lower()}" if random.random() < 0.5 else f"{name.lower()}_"
    )

    def __call__(self, token: str) -> str:
        return self.value(token)


def _alter_token(token: str) -> str:
    return random.choice(list(TokenAlteringOperations))(token)


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


class IdentifierOperation(Operation, ABC):
    def is_applicable(self, token: Token, context: tuple[Token, Token]) -> bool:
        return isinstance(token, Identifier)


class IdentifierAlteringOperation(IdentifierOperation):
    """
    Operation altering an identifier in more intelligent way.
    """

    def _apply(self, code: str) -> str:
        operation = random.choice(list(IdentifierAlteringOperations))
        return operation(code)


class IdentifierRandomAlteringOperation(IdentifierOperation):
    """
    Operation altering an identifier randomly.
    """

    def _apply(self, code: str) -> str:
        sub_tokens = Identifier.sub_tokens_of_identifier(code)
        random_tokens = random.choices(sub_tokens)
        for idx, token in enumerate(sub_tokens):
            if token in random_tokens:
                sub_tokens[idx] = _alter_token(token)
        return "".join(sub_tokens)


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

    INSERT_SPACE = (InsertOperation(" "), 5)
    INSERT_TAB = (InsertOperation("\t"), 1)
    INSERT_NL = (InsertOperation("\n"), 5)
    # DELETE_TAB = DeleteOperation("\t") styler does not support
    DELETE_SPACE = (DeleteOperation(" "), 5)
    DELETE_NL = (DeleteOperation("\n"), 5)
    NAME_ALTERING = (IdentifierRandomAlteringOperation(), 8)

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
            OperationNonApplicableException,
        ),
    )
    @returns_valid_java
    def __generate_violation(self, tokens: list[Token]) -> str:
        """
        This is the actual method called to generate the violation.
        It ensures that the code is parseable.
        Also, the generation is tried 3 times if one of the above regulations
        is violated or the OperationNotApplicableException is thrown.
        :param tokens: Tokens of the non violated file.
        :return: Returns violated Java code.
        """

        # Copy tokens to ensure only one altering in each file.
        copied_tokens = copy.deepcopy(tokens)
        return self._generate_violation(copied_tokens)

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
        Generates the violations.
        It ensures that the generated violations are valid java code and
        that they contain exactly one violation in each sample.
        The generation is done in batches of DEFAULT_GEN_BATCH_SIZE, as the
        most time-consuming step is the setup of the external checkstyle process.
        Therefore, it is way more efficient to call checkstyle on multiple samples and
        throw the invalid ones away.
        :return:
        """
        self._preprocessing_steps()
        os.makedirs(self.save_path, exist_ok=True)
        with tqdm(total=self.n, desc="Generate violations") as progress_bar:
            start = time.time()
            valid_violations = 0
            while valid_violations < self.n and time.time() - start < self.delta:
                valid_violations = self._batched_violation_generation(valid_violations)
                progress_bar.update(valid_violations - progress_bar.n)
        self._postprocessing_steps()
        self._generate_metadata()

    def _batched_violation_generation(self, valid_violations_count) -> int:
        """
        As the overhead of starting an external process it is better to
        create batches and run checkstyle on this batch and throw away
        invalid samples.
        :return: Returns amount of already valid generated violation.
        """

        # Sample at most remaining violations
        batch_size = min(DEFAULT_GEN_BATCH_SIZE, self.n - valid_violations_count)
        valid_pairs: list[(Path, (str, str))] = []

        # Ensure full batch on which checkstyle is run,
        # as this is the most time-consuming step.
        while len(valid_pairs) < batch_size:
            # As self.__generate_violations ensures parseable java code skip
            # examples which are not parseable and where the selected
            # operations cannot be applied.
            with suppress(TooManyTriesException):
                current_file = random.choice(self.non_violated_sources)
                content = read_content_of_file(current_file)
                tokens = tokenize_java_code(content)
                non_violated = "".join(
                    stream(tokens).map(lambda token: token.de_tokenize())
                )
                violated = self.__generate_violation(tokens)
                valid_pairs.append((current_file, (non_violated, violated)))

        # Initialized input for run_checkstyle_on_strs
        # id == index in valid_pairs which later is used to filter
        # generated strs with exactly 1 violation.
        violated_dict = dict(
            stream(valid_pairs).map(lambda p: p[1][1]).enumerate().to_dict()
        )

        # Checkstyle report cannot be parsed skip the batch.
        with suppress(ParseError):
            reports_with_id = run_checkstyle_on_strs(
                violated_dict, self.checkstyle_version, self.checkstyle_config
            )

            # Get generated instances with exactly 1 violation
            valid_violations: list[(Path, (str, str))] = []
            for idx, report in reports_with_id.items():
                if len(report.violations) == 1:
                    valid_violations.append(valid_pairs[idx])

            # Save generated violations
            for current_file, (violated, non_violated) in valid_violations:
                current_save_path = self.save_path / Path(str(valid_violations_count))
                os.makedirs(current_save_path, exist_ok=True)
                non_violated_file_name = Path(current_file.name)
                violated_file_name = Path(f"VIOLATED_{current_file.name}")
                save_content_to_file(
                    current_save_path / non_violated_file_name, non_violated
                )
                save_content_to_file(current_save_path / violated_file_name, violated)
                valid_violations_count += 1

        return valid_violations_count

    def _generate_metadata(self) -> None:
        with tqdm(total=self.n, desc="Generate metadata") as progress_bar:
            for sample_dir in os.listdir(self.save_path):
                curr_dir = self.save_path / Path(sample_dir)
                for _, _, files in os.walk(curr_dir):
                    violated, non_violated = None, None
                    for file in files:
                        if file.startswith("VIOLATED"):
                            violated = curr_dir / Path(file)
                        elif file.endswith(".java"):
                            non_violated = curr_dir / Path(file)
                    progress_bar.update()
                    try:
                        metadata = Metadata(
                            non_violated,
                            violated,
                            self.checkstyle_config,
                            self.checkstyle_version,
                        )
                        metadata.save_to_directory(curr_dir)
                    except MetadataException:
                        shutil.rmtree(curr_dir)


class RandomGenerator(ViolationGenerator):
    """
    Random violations generator.
    """

    def _preprocessing_steps(self) -> None:
        pass

    def _postprocessing_steps(self) -> None:
        pass

    def _generate_violation(self, tokens: list[Token]) -> str:
        operation = random.choices(
            [op.value[0] for op in list(Operations)],
            k=1,
            weights=[op.value[1] for op in list(Operations)],
        )[0]
        applicable_tokens = self._get_applicable_tokens(operation, tokens)
        if not applicable_tokens:
            raise OperationNonApplicableException(
                f"{operation} not applicable on token sequence."
            )
        random_token = random.choice(applicable_tokens)
        old_value_of_token = random_token.text
        operation(random_token)
        new_value_of_token = random_token.text
        if operation == Operations.NAME_ALTERING:
            return "".join(
                stream(tokens).map(lambda token: token.de_tokenize())
            ).replace(old_value_of_token, new_value_of_token)
        return "".join(stream(tokens).map(lambda token: token.de_tokenize()))

    @staticmethod
    def _get_applicable_tokens(
        operation: Operation, tokens: list[Token]
    ) -> list[Token]:
        if operation == Operations.NAME_ALTERING:
            return list(
                stream(tokens)
                .filter(lambda token: isinstance(token, Identifier))
                .to_list()
            )

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
    token_before: str
    whitespace: str
    token_after: str
    count: int

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ThreeGram):
            return False
        return (
            self.token_before == other.token_before
            and self.whitespace == other.whitespace
            and self.token_after == other.token_after
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __repr__(self) -> str:
        return (
            f"[{self.token_before}, "
            f"{self.whitespace}, "
            f"{self.token_after}, "
            f"{self.count}]"
        )


class ThreeGramGenerator(ViolationGenerator):
    """
    3-gram violation generator.
    """

    def _preprocessing_steps(self) -> None:
        self.__load_three_grams()

    def _postprocessing_steps(self) -> None:
        pass

    def _generate_violation(self, tokens: list[Token]) -> str:
        if random.random() < 0.1:
            return self._generate_name_violations(tokens)
        three_grams = list(self.__build_3_grams_from_tokens(tokens))
        if not three_grams:
            raise OperationNonApplicableException("Cannot generate on empty seq.")
        three_gram = random.choice(three_grams)
        alternatives_with_prob = self.__get_alternatives_with_prob(three_gram[0])
        if not alternatives_with_prob:
            raise OperationNonApplicableException
        weights = list(alternatives_with_prob.values())
        choices = list(
            stream(list(alternatives_with_prob.keys())).map(
                lambda t_gram: t_gram.whitespace
            )
        )
        new_ws = random.choices(choices, weights)
        three_gram[1].text = Whitespace.parse_tokenized_str(new_ws[0])
        return "".join(stream(tokens).map(lambda token: token.de_tokenize()))

    @staticmethod
    def _generate_name_violations(tokens: list[Token]) -> str:
        altering_operation = IdentifierAlteringOperation()
        applicable_tokens = list(filter(lambda t: isinstance(t, Identifier), tokens))
        if not applicable_tokens:
            raise OperationNonApplicableException(
                f"{altering_operation} not applicable on token sequence."
            )
        random_token = random.choice(applicable_tokens)
        old_value_of_token = random_token.text
        altering_operation(random_token)
        new_value_of_token = random_token.text
        return "".join(stream(tokens).map(lambda token: token.de_tokenize())).replace(
            old_value_of_token, new_value_of_token
        )

    def __load_three_grams(self) -> None:
        if not CSV_PATH.exists():
            raise FileNotFoundError("The csv/three_grams.csv does not exist.")

        self.three_grams = list(self.__build_3_grams_from_csv())

    @staticmethod
    def __build_3_grams_from_csv() -> Generator[ThreeGram, Any, None]:
        with open(CSV_PATH) as three_grams_file:
            csv_reader = csv.reader(three_grams_file)

            # Skip header
            next(csv_reader, None)
            for row in csv_reader:
                yield ThreeGram(row[0], row[1], row[2], int(row[3]))

    @staticmethod
    def __build_3_grams_from_tokens(
        tokens: list[Token],
    ) -> Generator[(ThreeGram, Whitespace), Any, None]:
        whitespaces = (
            stream(tokens).filter(lambda token: isinstance(token, Whitespace)).to_list()
        )
        for whitespace in whitespaces:
            idx = tokens.index(whitespace)
            token_before = None if idx == 0 else tokens[idx - 1]
            token_after = None if idx == len(tokens) - 1 else tokens[idx + 1]
            yield ThreeGram(
                str(token_before), str(whitespace), str(token_after), -1
            ), whitespace

    def __get_alternatives_with_prob(
        self, three_gram: ThreeGram
    ) -> dict[ThreeGram, float]:
        alternatives = (
            stream(self.three_grams)
            .filter(lambda tg: tg != three_gram)
            .filter(
                lambda tg: tg.token_before == three_gram.token_before
                and tg.token_after == three_gram.token_after
            )
            .to_list()
        )
        total = alternatives.map(lambda entry: entry.count).reduce(
            lambda entry1, entry2: entry1 + entry2, 0
        )
        return dict(
            alternatives.map(lambda entry: (entry, entry.count / total)).to_dict()
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
        violation_type: str | None = None,
    ) -> None:
        self.non_violated_source = non_violated_source
        self.violated_source = violated_source
        self.config = config
        self.version = version
        if not non_violated_str or not violated_str or not violation_type:
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
            "violated_str": self.violated_str,
            "violation_type": self.violation_type,
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
            data["violation_type"],
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
        violated, non_violated = None, None
        for processed_file in processed_files:
            if len(processed_file.report.violations) == 0:
                non_violated = processed_file
            elif len(processed_file.report.violations) == 1:
                self.violation_type = next(
                    iter(processed_file.report.violations)
                ).type.value
                violated = processed_file
        if not non_violated or not violated or not self.violation_type:
            raise MetadataException("Violation amount wrong!")
        self.non_violated_str, self.violated_str = filter_relevant_tokens(
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
    delta: int = 4 * 60 * 60,  # 4h
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


def filter_relevant_tokens(
    non_violated: ProcessedSourceFile, violated: ProcessedSourceFile, context: int = 2
) -> (str, str):
    """
    Filter the relevant tokens out of the non_violated and violated source file.
    :param non_violated: The not violated source file.
    :param violated: The violated source file.
    :param context: The specified line context around the violation.
    :return: Returns the filtered non_violated and violated source tokens.
    """
    assert len(violated.checkstyle_tokens) == 2
    violated_tokens = next(violated.tokens_between_violations())
    violated_tokens_wo_checkstyle = (
        stream(violated.tokens)
        .filter(lambda token: not isinstance(token, CheckstyleToken))
        .to_list()
    )
    start_idx_non_violated = violated_tokens_wo_checkstyle.index(violated_tokens[1])
    end_idx_non_violated = violated_tokens_wo_checkstyle.index(violated_tokens[-2]) + 1
    non_violated_tokens = non_violated.tokens[
        start_idx_non_violated:end_idx_non_violated
    ]
    violation_type = next(iter(violated.report.violations)).type
    interesting_token_type = interesting_tokens_type(violation_type)
    return (
        " ".join(
            stream(non_violated_tokens)
            # TODO: styler uses only whitespaces but we want to use all tokens
            #       (knowingly that might decrease the performance)
            .filter(lambda t: isinstance(t, interesting_token_type)).map(str)
        ),
        " ".join(stream(next(violated.violations_with_ctx(context))).map(str)),
    )
