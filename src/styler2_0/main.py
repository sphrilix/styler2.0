import argparse
import sys
from argparse import ArgumentParser, Namespace
from enum import Enum
from pathlib import Path

from src.styler2_0.utils.checkstyle import run_checkstyle_on_dir
from src.styler2_0.utils.tokenize import tokenize_dir, tokenize_with_reports
from src.styler2_0.utils.utils import enum_action
from src.styler2_0.utils.violation_generation import Protocol, generate_n_violations


class ArgumentMissingException(Exception):
    """
    Exception thrown if a required argument is missing.
    """


class Tasks(Enum):
    RUN_CHECKSTYLE = "RUN_CHECKSTYLE"
    GENERATE_VIOLATIONS = "GENERATE_VIOLATIONS"

    def __str__(self) -> str:
        return self.value


def main(args: list[str]) -> int:
    arg_parser = _set_up_arg_parser()
    parsed_args = arg_parser.parse_args(args)
    match parsed_args.task:
        case Tasks.GENERATE_VIOLATIONS:
            _run_violation_generation(parsed_args)
        case _:
            return 1
    return 0


def _run_violation_generation(parsed_args: Namespace) -> None:
    _check_presence_of_needed_args(
        parsed_args, ("protocol", "save", "source", "config", "version", "n")
    )
    protocol = parsed_args.protocol
    n = parsed_args.n
    save = Path(parsed_args.save)
    source = Path(parsed_args.source)
    config = Path(parsed_args.config)
    version = parsed_args.version

    generate_n_violations(n, protocol, source, config, version, save)


def _check_presence_of_needed_args(
    parsed_args: Namespace, args: tuple[str, ...]
) -> None:
    for arg in args:
        if arg not in parsed_args:
            raise ArgumentMissingException(
                f"When running task: {parsed_args.task} --{arg} must be specified."
            )


def _set_up_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--task", type=Tasks, choices=list(Tasks))

    # Set up arguments for generating violations
    arg_parser.add_argument(
        "--protocol",
        action=enum_action(Protocol),
        required=False,
    )
    arg_parser.add_argument("--n", type=int, required=False, default=argparse.SUPPRESS)
    arg_parser.add_argument("--save", required=False, default=argparse.SUPPRESS)
    arg_parser.add_argument("--source", required=False, default=argparse.SUPPRESS)
    arg_parser.add_argument("--config", required=False, default=argparse.SUPPRESS)
    arg_parser.add_argument("--version", required=False, default=argparse.SUPPRESS)
    return arg_parser


if __name__ == "__main__":
    print(tokenize_dir(Path("/Users/maxij/PycharmProjects/styler2.0/data")))
    # print(
    #     run_checkstyle_on_dir(
    #         Path("/Users/maxij/PycharmProjects/styler2.0/data"), "8.0"
    #     )
    # )
    reports = run_checkstyle_on_dir(
        Path("/Users/maxij/PycharmProjects/styler2.0/data"), "8.0"
    )
    print(sys.argv)
    main(sys.argv[1:])
    print(tokenize_with_reports(reports))
