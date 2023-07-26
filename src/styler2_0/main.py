import logging
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
    CHECKSTYLE_TIMEOUT,
    find_checkstyle_config,
    fix_checkstyle_config,
    run_checkstyle_on_dir,
)
from src.styler2_0.utils.maven import get_checkstyle_version_of_project
from src.styler2_0.utils.styler_adaption import adapt_styler_three_gram_csv
from src.styler2_0.utils.tested import extract_tested_src_files
from src.styler2_0.utils.utils import enum_action


class TaskNotSupportedException(Exception):
    """
    Exception is thrown whenever a task is not supported.
    """


class Tasks(Enum):
    GENERATE_VIOLATIONS = "GENERATE_VIOLATIONS"
    CHECKSTYLE = "CHECKSTYLE"
    ADAPT_THREE_GRAMS = "ADAPT_THREE_GRAMS"
    PREPROCESSING = "PREPROCESSING"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        raise TaskNotSupportedException(f"{value} is a not supported Task!")

    def __str__(self) -> str:
        return self.value


def setup_logging(log_file: str = "styler.log", overwrite: bool = False) -> None:
    """
    Set up logging.
    """
    # Get the overwrite flag
    mode = "w" if overwrite else "a"

    # Set the logging level
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level)

    # Create a file handler to write messages to a log file
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(logging_level)

    # Create a console handler to display messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)

    # Define the log format
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Get the root logger and add the handlers
    logger = logging.getLogger("")
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def main(args: list[str]) -> int:
    arg_parser = _set_up_arg_parser()
    parsed_args = arg_parser.parse_args(args)
    task = Tasks(parsed_args.command)

    # Set up logging and specify logfile name
    logfile = "styler.log"
    if parsed_args.save:
        folder_path = Path(parsed_args.save)
        folder_name = Path(parsed_args.save).name
        logfile = folder_path / Path(f"styler-{folder_name}.log")
    setup_logging(logfile, overwrite=True)

    match task:
        case Tasks.GENERATE_VIOLATIONS:
            _run_violation_generation(parsed_args)
        case Tasks.CHECKSTYLE:
            _run_checkstyle_report(parsed_args)
        case Tasks.ADAPT_THREE_GRAMS:
            adapt_styler_three_gram_csv(parsed_args.in_file, parsed_args.out_file)
        case Tasks.PREPROCESSING:
            preprocessing(parsed_args.violation_dir, parsed_args.splits)
        case _:
            return 1
    return 0


def _run_checkstyle_report(parsed_args: Namespace):
    save = parsed_args.save
    source = parsed_args.source
    config = parsed_args.config
    tested = parsed_args.tested
    timeout = parsed_args.timeout

    os.makedirs(save, exist_ok=True)

    version = parsed_args.version
    if not version:
        try:
            version = get_checkstyle_version_of_project(source)
        except AttributeError as e:
            # version = find_version_by_trying(config, source)
            raise Exception("No version found!") from e
    logging.info(f"Checkstyle version: {version}")

    if not config:
        config = find_checkstyle_config(source)
        fix_checkstyle_config(config, save / Path("checkstyle-fixed.xml"), version)
        config = save / Path("checkstyle-fixed.xml")
    logging.info(f"Checkstyle config: {config}")

    checkstyle_report = run_checkstyle_on_dir(source, version, config, timeout)
    logging.info("Checkstyle report created.")

    non_violated_files = (
        stream(list(checkstyle_report))
        .filter(lambda report: not report.violations)
        .map(lambda report: report.path)
        .to_set()
    )
    logging.info(f"Non violated files extracted. Amount: {len(non_violated_files)}")

    # remove all not tested files and all test files
    if tested:
        non_violated_files = extract_tested_src_files(non_violated_files)
        logging.info(f"Tested files extracted. Amount: {len(non_violated_files)}")

    non_violated_dir = save / Path("non_violated/")
    os.makedirs(non_violated_dir, exist_ok=True)

    for file in non_violated_files:
        with suppress(SameFileError):
            copyfile(file, non_violated_dir / file.name)

    if tested:
        logging.info(f"Non violated and tested files stored in {non_violated_dir}")
    else:
        logging.info(f"Non violated files stored in {non_violated_dir}")

    return non_violated_dir, version


def _run_violation_generation(parsed_args: Namespace) -> None:
    protocol = parsed_args.protocol
    n = parsed_args.n
    save = parsed_args.save
    config = parsed_args.config

    non_violated_dir, version = _run_checkstyle_report(parsed_args)

    violations_dir = save / Path("violations")
    os.makedirs(violations_dir, exist_ok=True)

    generate_n_violations(
        n, protocol, non_violated_dir, config, version, violations_dir
    )


def _set_up_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    sub_parser = arg_parser.add_subparsers(dest="command", required=True)
    generation = sub_parser.add_parser(str(Tasks.GENERATE_VIOLATIONS))
    adapting_three_gram = sub_parser.add_parser(str(Tasks.ADAPT_THREE_GRAMS))
    preprocessing_sub_parser = sub_parser.add_parser(str(Tasks.PREPROCESSING))
    checkstyle_sub_parser = sub_parser.add_parser(str(Tasks.CHECKSTYLE))

    # Set up arguments for generating violations
    _add_checkstyle_arguments(generation)
    generation.add_argument(
        "--protocol",
        action=enum_action(Protocol),
        required=False,
    )
    generation.add_argument("--n", type=int, required=True)
    generation.add_argument("--delta", required=False, type=int, default=10800)

    # Set up arguments for checkstyle
    _add_checkstyle_arguments(checkstyle_sub_parser)

    # Set up arguments for adapting styler csv
    adapting_three_gram.add_argument("--in_file", type=Path, required=True)
    adapting_three_gram.add_argument("--out_file", type=Path, required=True)

    # Set up arguments for model preprocessing
    preprocessing_sub_parser.add_argument("--violation_dir", type=Path, required=True)
    preprocessing_sub_parser.add_argument(
        "--splits", type=tuple[float, float, float], default=(0.9, 0.1, 0.0)
    )

    return arg_parser


def _add_checkstyle_arguments(checkstyle_sub_parser):
    checkstyle_sub_parser.add_argument("--save", required=True, type=Path)
    checkstyle_sub_parser.add_argument("--source", required=True, type=Path)
    checkstyle_sub_parser.add_argument(
        "--config", required=False, type=Path, default=None
    )
    checkstyle_sub_parser.add_argument("--version", required=False)
    checkstyle_sub_parser.add_argument("--tested", action="store_true")
    checkstyle_sub_parser.add_argument(
        "--timeout", required=False, type=int, default=CHECKSTYLE_TIMEOUT
    )


if __name__ == "__main__":
    main(sys.argv[1:])
