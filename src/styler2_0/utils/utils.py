import argparse
import json
import os
from argparse import Action
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import yaml
from chardet import detect
from yaml import SafeLoader

T = TypeVar("T")
E = TypeVar("E", bound=Enum)


def enum_action(enum_cls: type[E]) -> type[Action]:
    """
    Casting enums in ArgParser based on their name.
    :param enum_cls: The type of the enum.
    :return: Returns the Action to perform casting.
    """

    class EnumAction(Action):
        """
        Action to parse enums based on their name.
        """

        def __init__(
            self,
            option_strings: Sequence[str],
            dest: str,
            nargs: str | int | None = None,
            const: E | None = None,
            default: E | str | None = None,
            type: Callable[[str], E] | argparse.FileType | None = None,
            required: bool = False,
            help_str: str | None = None,
            metavar: str | tuple[str, ...] | None = None,
        ) -> None:
            if isinstance(default, str):
                default = enum_cls[default.upper()]
            if default is not None:
                if help_str is None:
                    help_str = f"(default: {default.name.lower()})"
                else:
                    help_str = f"{help_str} (default: {default.name.lower()})"
            self.cls = enum_cls
            super().__init__(
                option_strings,
                dest,
                nargs=nargs,
                const=const,
                default=default,
                type=type,
                choices=[variant.name for variant in enum_cls],  # type: ignore
                required=required,
                help=help_str,
                metavar=metavar,
            )

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: str | Sequence[Any] | None = None,
            option_str: None | str = None,
        ) -> None:
            if not isinstance(values, str):
                raise TypeError
            setattr(namespace, self.dest, getattr(self.cls, values.upper()))

    return EnumAction


class TooManyTriesException(Exception):
    """
    If a function is decorated with @retry and does not pass a successful run within
    the specified amount this exception is thrown.
    """


def retry(
    n: int,
    default: T = None,
    exceptions: type[Exception] | tuple[type[Exception], ...] = Exception,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator, that retries the decorated function n times. If within those n
    steps no successful run occurred it raises a TooManyTriesException or if a
    default is given that will be returned.
    Also, the decorator only catches the exceptions from the given ones.
    :param n: The given tries.
    :param default: The given default.
    :param exceptions: The exceptions to be caught.
    :return: The decorated function.
    """

    def _retry_decorator(func: Callable[..., T]) -> Callable[..., T]:
        def _retry_decorator_wrapper(*args: ..., **kwargs: ...) -> T:
            thrown_exceptions: list[str] = []
            for _ in range(n):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    thrown_exceptions.append(type(e).__name__)
            if default:
                return default
            raise TooManyTriesException(
                f"Within {n} tries no successful run of "
                f"{getattr(func, '__name__', repr(callable))} "
                f"(Exceptions raised: [{', '.join(thrown_exceptions)}])."
            )

        return _retry_decorator_wrapper

    return _retry_decorator


def save_content_to_file(file: Path, content: str) -> None:
    """
    Saves the given content to the specified file.
    :param file: The given file.
    :param content: The given content.
    :return: None
    """
    with open(file, "w", encoding="utf-8") as file_stream:
        file_stream.write(content)


def read_content_of_file(file: Path, encoding: str = None) -> str:
    """
    Read the content of a file to str.
    :param file: The given file.
    :param encoding: The given encoding.
    :return: Returns the file content as str.
    """
    if not encoding:
        encoding = _get_encoding_type(file)

    with open(file, encoding=encoding) as file_stream:
        return file_stream.read()


def _get_encoding_type(file: Path) -> str:
    with open(file, "rb") as f:
        rawdata = f.read()
    return detect(rawdata)["encoding"]


def get_files_in_dir(directory: Path, suffix: str = None) -> list[Path]:
    """
    Get files in given directory with the given suffix.
    :param directory: The given directory.
    :param suffix: The given suffix.
    :return: Returns the files matching the suffix.
    """
    files_in_dir: list[Path] = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if not suffix or file.endswith(suffix):
                files_in_dir.append(Path(subdir) / Path(file))

    return files_in_dir


def get_sub_dirs_in_dir(directory: Path, depth: int = 1) -> list[Path]:
    """
    Get subdirectories in given directory with the given depth.
    Set depth = -1 to get all subdirectories.
    :param directory: The given directory.
    :param depth: The given depth.
    :return: Returns the subdirectories.
    """
    sub_dirs_in_dir: list[Path] = []
    for subdir, _, _ in os.walk(directory):
        subdir = Path(subdir)
        if subdir != directory and (
            depth == -1
            or str(subdir).count(os.path.sep) - str(directory).count(os.path.sep)
            <= depth
        ):
            sub_dirs_in_dir.append(Path(subdir))

    return sub_dirs_in_dir


def load_yaml_file(path: Path) -> dict[str, Any]:
    """
    Loads a yaml file to a dict.
    :param path: The path to the yaml file.
    :return: Returns the loaded yaml as dict.
    """
    raw_str = read_content_of_file(path)
    return yaml.load(raw_str, Loader=SafeLoader)


def collect_git_pre_training_data(projects_dir: Path, save: Path) -> None:
    """
    Collect pretraining data from mined violations.
    :param projects_dir: The directory where the mined repos are.
    :param save: Directory to save the data.
    :return:
    """
    projects = get_sub_dirs_in_dir(projects_dir)
    count = 0
    for project in projects:
        mined_vios = project / "mined_violations"
        checkstyle_data = json.loads(read_content_of_file(mined_vios / "data.json"))
        cs_version = checkstyle_data["version"]
        cs_conf = checkstyle_data["config"]
        for vio_dir in get_sub_dirs_in_dir(mined_vios):
            vio_json = json.loads(read_content_of_file(vio_dir / "data.json"))
            if "fix_str" in vio_json and "violation_str" in vio_json:
                non_violated_src = next(iter(get_files_in_dir(vio_dir / "violation")))
                violated_src = next(iter(get_files_in_dir(vio_dir / "fix")))
                vio_type = vio_json["violation_type"]
                violated_str = vio_json["violation_str"]
                non_violated_str = vio_json["fix_str"]

                # Skip empty training samples.
                if not (violated_str and non_violated_str):
                    continue
                vio_save = save / "violations/pre_training" / str(count)
                os.makedirs(vio_save, exist_ok=True)
                save_content_to_file(
                    vio_save / "data.json",
                    json.dumps(
                        {
                            "violated_str": violated_str,
                            "non_violated_str": non_violated_str,
                            "version": cs_version,
                            "config": cs_conf,
                            "non_violated_source": str(non_violated_src),
                            "violated_source": str(violated_src),
                            "violation_type": vio_type,
                        }
                    ),
                )
                count += 1
