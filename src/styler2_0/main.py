import os
import sys
from argparse import ArgumentParser, Namespace
from contextlib import suppress
from enum import Enum
from pathlib import Path
from shutil import SameFileError, copyfile
from typing import Any

from streamerate import stream

from src.styler2_0.models.models import Models, evaluate, train
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
from src.styler2_0.utils.git_utils import (
    collect_git_pre_training_data,
    process_git_repository,
)
from src.styler2_0.utils.styler_adaption import adapt_styler_three_gram_csv
from src.styler2_0.utils.utils import enum_action
from styler2_0.utils.analysis import (
    analyze_all_eval_jsons,
    analyze_data_dir,
    analyze_generated_violations,
)


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
    EVAL = "EVAL"
    ANALYZE_EVAL = "ANALYZE_EVAL"
    ANALYZE_DIR = "ANALYZE_DIR"
    PRE_TRAINING = "PRE_TRAINING"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        raise TaskNotSupportedException(f"{value} is a not supported Task!")

    def __str__(self) -> str:
        return self.value


def _run_evaluation(
    model: Models,
    project_dir: Path,
    mined_violation_dir: Path,
    top_k: int,
) -> None:
    evaluate(model, mined_violation_dir, project_dir, top_k)


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
            _run_preprocessing(parsed_args)
        case Tasks.MINE_VIOLATIONS:
            _run_violation_mining(parsed_args.repo, parsed_args.save)
        case Tasks.TRAIN:
            _run_train(parsed_args)
        case Tasks.EVAL:
            _run_evaluation(
                parsed_args.model,
                parsed_args.project_dir,
                parsed_args.mined_violations_dir,
                parsed_args.top_k,
            )
        case Tasks.ANALYZE_EVAL:
            analyze_all_eval_jsons(parsed_args.eval_dir)
        case Tasks.ANALYZE_DIR:
            analyze_data_dir(parsed_args.dir)
        case Tasks.PRE_TRAINING:
            _run_pre_training(parsed_args)
        case _:
            return 1
    return 0


def _run_preprocessing(parsed_args: Namespace) -> None:
    preprocessing(
        parsed_args.project_dir,
        parsed_args.splits,
        parsed_args.model,
        parsed_args.src_vocab,
        parsed_args.trg_vocab,
    )


def _run_violation_generation(parsed_args: Namespace) -> None:
    protocol = parsed_args.protocol
    n = parsed_args.n
    save = parsed_args.save
    source = parsed_args.source
    config = parsed_args.config
    version = parsed_args.version

    os.makedirs(save, exist_ok=True)

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
    analyze_generated_violations(violations_dir)


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
    """
    Tries to find the checkstyle version of the project.
    If it fails, it tries to find the version by trying.
    :param input_dir: The directory of the project.
    :param config: The checkstyle config.
    :return: Returns the checkstyle version.
    """
    # try:
    #     return get_checkstyle_version_of_project(input_dir)
    # except AttributeError:
    return find_version_by_trying(config, input_dir)


def _run_violation_mining(repo_dir: Path, save: Path) -> None:
    """
    Runs the violation mining on the given repository.
    :param repo_dir: The directory of the repository.
    :param save: The directory where the results should be saved.
    :return:
    """
    config = find_checkstyle_config(repo_dir)
    version = _get_checkstyle_version(repo_dir, config)
    process_git_repository(repo_dir, save, version, config)


def _run_pre_training(parsed_args: Namespace) -> None:
    """
    Runs the pre-training.
    :param parsed_args: The parsed args for pre-training.
    :return:
    """
    collect_git_pre_training_data(parsed_args.projects_dir, parsed_args.save)
    for model in Models:
        preprocessing(
            parsed_args.save,
            parsed_args.splits,
            model,
            parsed_args.save
            / f"model_data/pre_training/{model.name.lower()}/src_vocab.txt",
            parsed_args.save
            / f"model_data/pre_training/{model.name.lower()}/trg_vocab.txt",
        )
        train(model, parsed_args.save / "model_data", parsed_args.epochs)


def _run_train(parsed_args: Namespace) -> None:
    if not parsed_args.from_pretrained:
        train(parsed_args.model, parsed_args.path, parsed_args.epochs, parsed_args.lr)
    else:
        train(
            parsed_args.model,
            parsed_args.path,
            parsed_args.epochs,
            parsed_args.lr,
            parsed_args.from_pretrained,
        )


def _set_up_arg_parser() -> ArgumentParser:
    """
    Sets up the argument parser.
    :return: Returns the argument parser.
    """
    arg_parser = ArgumentParser()
    sub_parser = arg_parser.add_subparsers(dest="command", required=True)
    generation = sub_parser.add_parser(str(Tasks.GENERATE_VIOLATIONS))
    adapting_three_gram = sub_parser.add_parser(str(Tasks.ADAPT_THREE_GRAMS))
    preprocessing_sub_parser = sub_parser.add_parser(str(Tasks.PREPROCESSING))
    mine_violations_sub_parser = sub_parser.add_parser(str(Tasks.MINE_VIOLATIONS))
    train_sub_parser = sub_parser.add_parser(str(Tasks.TRAIN))
    eval_sub_parser = sub_parser.add_parser(str(Tasks.EVAL))
    analyze_eval_sub_parser = sub_parser.add_parser(str(Tasks.ANALYZE_EVAL))
    analyze_dir_sub_parser = sub_parser.add_parser(str(Tasks.ANALYZE_DIR))
    pre_training_sub_parser = sub_parser.add_parser(str(Tasks.PRE_TRAINING))

    # Set up arguments for generating violations
    generation.add_argument(
        "--protocol",
        action=enum_action(Protocol),
        required=True,
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
    preprocessing_sub_parser.add_argument("--project_dir", type=Path, required=True)
    preprocessing_sub_parser.add_argument(
        "--splits", type=tuple[float, float, float], default=(0.9, 0.1, 0.0)
    )
    preprocessing_sub_parser.add_argument(
        "--model", action=enum_action(Models), required=True
    )
    preprocessing_sub_parser.add_argument("--src_vocab", type=Path, default=None)
    preprocessing_sub_parser.add_argument("--trg_vocab", type=Path, default=None)

    # Set up arguments for model training
    train_sub_parser.add_argument("--model", action=enum_action(Models), required=True)
    train_sub_parser.add_argument("--path", type=Path, required=True)
    train_sub_parser.add_argument("--epochs", type=int, required=True)
    train_sub_parser.add_argument("--lr", type=float, default=1e-3)
    train_sub_parser.add_argument("--from_pretrained", type=Path, default=None)

    # Set up arguments for mining violations
    mine_violations_sub_parser.add_argument("--repo", type=Path, required=True)
    mine_violations_sub_parser.add_argument("--save", type=Path, required=True)

    # Set up arguments for evaluation
    eval_sub_parser.add_argument("--model", action=enum_action(Models), required=True)
    eval_sub_parser.add_argument("--project_dir", type=Path, required=True)
    eval_sub_parser.add_argument("--top_k", type=int, default=5)
    eval_sub_parser.add_argument("--mined_violations_dir", type=Path, required=True)

    # Set up arguments for analyzing evaluation
    analyze_eval_sub_parser.add_argument("--eval_dir", type=Path, required=True)

    # Set up arguments for analyzing directory
    analyze_dir_sub_parser.add_argument("--dir", type=Path, required=True)

    # Set up arguments for pre-training
    pre_training_sub_parser.add_argument("--projects_dir", required=True, type=Path)
    pre_training_sub_parser.add_argument("--save", required=True, type=Path)
    pre_training_sub_parser.add_argument(
        "--splits", type=tuple[float, float, float], default=(0.9, 0.1, 0.0)
    )
    pre_training_sub_parser.add_argument("--epochs", type=int, required=True)

    return arg_parser


if __name__ == "__main__":
    main()
