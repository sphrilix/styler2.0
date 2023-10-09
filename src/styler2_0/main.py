import os
import sys
from argparse import ArgumentParser, Namespace
from contextlib import suppress
from enum import Enum
from pathlib import Path
from shutil import SameFileError, copyfile
from typing import Any

from streamerate import stream

from src.styler2_0.preprocessing.model_preprocessing import preprocessing
from src.styler2_0.preprocessing.violation_generation import (
    Protocol,
    generate_n_violations,
)
from src.styler2_0.utils.checkstyle import (
    find_checkstyle_config,
    find_version_by_trying,
    run_checkstyle_on_dir,
)
from src.styler2_0.utils.git_utils import process_git_repository
from src.styler2_0.utils.maven import get_checkstyle_version_of_project
from src.styler2_0.utils.styler_adaption import adapt_styler_three_gram_csv
from src.styler2_0.utils.utils import enum_action
from styler2_0.models.models import Models, train


class TaskNotSupportedException(Exception):
    """
    Exception is thrown whenever a task is not supported.
    """


class Tasks(Enum):
    GENERATE_VIOLATIONS = "GENERATE_VIOLATIONS"
    ADAPT_THREE_GRAMS = "ADAPT_THREE_GRAMS"
    PREPROCESSING = "PREPROCESSING"
    TRAIN = "TRAIN"
    MINE_VIOLATIONS = "MINE_VIOLATIONS"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        raise TaskNotSupportedException(f"{value} is a not supported Task!")

    def __str__(self) -> str:
        return self.value


def main() -> int:
    args = sys.argv[1:]
    arg_parser = _set_up_arg_parser()
    parsed_args = arg_parser.parse_args(args)
    task = Tasks(parsed_args.command)
    match task:
        case Tasks.GENERATE_VIOLATIONS:
            _run_violation_generation(parsed_args)
        case Tasks.ADAPT_THREE_GRAMS:
            adapt_styler_three_gram_csv(parsed_args.in_file, parsed_args.out_file)
        case Tasks.PREPROCESSING:
            preprocessing(parsed_args.violation_dir, parsed_args.splits)
        case Tasks.MINE_VIOLATIONS:
            _run_violation_mining(parsed_args.repo, parsed_args.save)
        case Tasks.TRAIN:
            train(
                parsed_args.model,
                parsed_args.path,
                parsed_args.epochs,
            )
        case _:
            return 1
    return 0


def _run_violation_generation(parsed_args: Namespace) -> None:
    protocol = parsed_args.protocol
    n = parsed_args.n
    save = parsed_args.save
    source = parsed_args.source
    config = parsed_args.config
    version = parsed_args.version

    if not config:
        config = find_checkstyle_config(source)

    copyfile(config, save / Path("checkstyle.xml"))

    if not version:
        version = _get_checkstyle_version(source, config)

    os.makedirs(save, exist_ok=True)

    non_violated_files = _extract_non_violated_files(save, source, config, version)

    non_violated_dir = save / Path("non_violated/")
    os.makedirs(non_violated_dir, exist_ok=True)

    for file in non_violated_files:
        with suppress(SameFileError):
            copyfile(file, non_violated_dir / file.name)

    violations_dir = save / Path("violations")
    os.makedirs(violations_dir, exist_ok=True)

    generate_n_violations(
        n, protocol, non_violated_dir, config, version, violations_dir
    )


def _extract_non_violated_files(
    save: Path, source: Path, config: Path, version: str
) -> set[Path]:
    checkstyle_report = run_checkstyle_on_dir(source, version, config)
    return set(
        stream(list(checkstyle_report))
        .filter(lambda report: not report.violations)
        .map(lambda report: report.path)
        .to_set()
    )


def _get_checkstyle_version(input_dir: Path, config: Path) -> str:
    try:
        return get_checkstyle_version_of_project(input_dir)
    except AttributeError:
        return find_version_by_trying(config, input_dir)


def _run_violation_mining(repo_dir: Path, save: Path) -> None:
    config = find_checkstyle_config(repo_dir)
    version = _get_checkstyle_version(repo_dir, config)
    process_git_repository(repo_dir, save, version, config)


def _set_up_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    sub_parser = arg_parser.add_subparsers(dest="command", required=True)
    generation = sub_parser.add_parser(str(Tasks.GENERATE_VIOLATIONS))
    adapting_three_gram = sub_parser.add_parser(str(Tasks.ADAPT_THREE_GRAMS))
    preprocessing_sub_parser = sub_parser.add_parser(str(Tasks.PREPROCESSING))
    mine_violations_sub_parser = sub_parser.add_parser(str(Tasks.MINE_VIOLATIONS))
    train_sub_parser = sub_parser.add_parser(str(Tasks.TRAIN))

    # Set up arguments for generating violations
    generation.add_argument(
        "--protocol",
        action=enum_action(Protocol),
        required=False,
    )
    generation.add_argument("--n", type=int, required=True)
    generation.add_argument("--save", required=True, type=Path)
    generation.add_argument("--source", required=True, type=Path)
    generation.add_argument("--config", required=False, type=Path, default=None)
    generation.add_argument("--version", required=False)
    generation.add_argument("--delta", required=False, type=int, default=10800)

    # Set up arguments for adapting styler csv
    adapting_three_gram.add_argument("--in_file", type=Path, required=True)
    adapting_three_gram.add_argument("--out_file", type=Path, required=True)

    # Set up arguments for model preprocessing
    preprocessing_sub_parser.add_argument("--violation_dir", type=Path, required=True)
    preprocessing_sub_parser.add_argument(
        "--splits", type=tuple[float, float, float], default=(0.9, 0.1, 0.0)
    )

    # Set up arguments for model training
    train_sub_parser.add_argument("--model", action=enum_action(Models), required=True)
    train_sub_parser.add_argument("--path", type=Path, required=True)
    train_sub_parser.add_argument("--epochs", type=int, required=True)

    # Set up arguments for mining violations
    mine_violations_sub_parser.add_argument("--repo", type=Path, required=True)
    mine_violations_sub_parser.add_argument("--save", type=Path, required=True)

    return arg_parser


if __name__ == "__main__":
    main()
