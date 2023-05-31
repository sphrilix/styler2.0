import sys
from argparse import ArgumentParser, Namespace
from enum import Enum
from pathlib import Path
from typing import Any

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
    save = Path(parsed_args.save)
    source = Path(parsed_args.source)
    config = Path(parsed_args.config)
    version = parsed_args.version

    generate_n_violations(n, protocol, source, config, version, save)


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
    generation.add_argument("--save", required=True)
    generation.add_argument("--source", required=True)
    generation.add_argument("--config", required=True)
    generation.add_argument("--version", required=True)
    generation.add_argument("--delta", required=False, type=int, default=10800)

    return arg_parser


if __name__ == "__main__":
    main(sys.argv[1:])
