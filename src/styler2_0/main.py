import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path

from utils.checkstyle import run_checkstyle_on_dir
from utils.tokenize import tokenize_dir, tokenize_with_reports


class Tasks(Enum):
    RUN_CHECKSTYLE = "RUN_CHECKSTYLE"
    GENERATE_VIOLATIONS = "GENERATE_VIOLATIONS"

    def __str__(self) -> str:
        return self.value


def _main(args: list[str]) -> int:
    arg_parser = _set_up_arg_parser()
    parsed_args = arg_parser.parse_args(args)
    print(parsed_args.task)
    return 0


def _set_up_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser("")
    arg_parser.add_argument("--task", type=Tasks, choices=list(Tasks))
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
    _main(sys.argv[1:])
    print(tokenize_with_reports(reports))
