import argparse
import os
from argparse import Action
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import yaml
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


def read_content_of_file(file: Path, encoding: str = "utf-8") -> str:
    """
    Read the content of a file to str.
    :param file: The given file.
    :param encoding: The given encoding.
    :return: Returns the file content as str.
    """
    with open(file, encoding=encoding) as file_stream:
        return file_stream.read()


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


def load_yaml_file(path: Path) -> dict[str, Any]:
    raw_str = read_content_of_file(path)
    return yaml.load(raw_str, Loader=SafeLoader)
