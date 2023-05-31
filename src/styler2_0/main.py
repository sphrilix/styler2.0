import os
import sys
from argparse import ArgumentParser, Namespace
from contextlib import suppress
from enum import Enum
from pathlib import Path
from shutil import SameFileError, copyfile
from typing import Any

from streamerate import stream

from src.styler2_0.utils.checkstyle import run_checkstyle_on_dir
from src.styler2_0.utils.maven import get_checkstyle_version_of_project
from src.styler2_0.utils.utils import enum_action
from src.styler2_0.utils.violation_generation import Protocol, generate_n_violations


class TaskNotSupportedException(Exception):
    """
    Exception is thrown whenever a task is not supported.
    """


class Tasks(Enum):
    RUN_CHECKSTYLE = "RUN_CHECKSTYLE"
    GENERATE_VIOLATIONS = "GENERATE_VIOLATIONS"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        raise TaskNotSupportedException(f"{value} is a not supported Task!")

    def __str__(self) -> str:
        return self.value


def main(args: list[str]) -> int:
    arg_parser = _set_up_arg_parser()
    parsed_args = arg_parser.parse_args(args)
    task = Tasks(parsed_args.command)
    match task:
        case Tasks.GENERATE_VIOLATIONS:
            _run_violation_generation(parsed_args)
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
    if not version:
        version = get_checkstyle_version_of_project(source)

    checkstyle_report = run_checkstyle_on_dir(source, version, config)

    non_violated_files = (
        stream(list(checkstyle_report))
        .filter(lambda report: not report.violations)
        .map(lambda report: report.path)
        .to_set()
    )

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


def _set_up_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    sub_parser = arg_parser.add_subparsers(dest="command", required=True)
    generation = sub_parser.add_parser(str(Tasks.GENERATE_VIOLATIONS))

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

    return arg_parser


if __name__ == "__main__":
    main(sys.argv[1:])
