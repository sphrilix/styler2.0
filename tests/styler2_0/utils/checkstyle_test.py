import os.path
import shutil
import tempfile
from pathlib import Path

import pytest

from src.styler2_0.utils.checkstyle import (
    WrongViolationAmountException,
    fix_checkstyle_config,
    returns_n_violations,
    run_checkstyle_on_dir,
    run_checkstyle_on_str,
)

CURR_DIR = os.path.dirname(os.path.relpath(__file__))
SAMPLE_PROJECT = os.path.join(CURR_DIR, "../../res/sample_project")
CHECKSTYLE_CONFIG = os.path.join(SAMPLE_PROJECT, "checkstyle.xml")

SAMPLE_PROJECT_3 = os.path.join(CURR_DIR, "../../res/sample_project_3")
CHECKSTYLE_CONFIG_3 = os.path.join(SAMPLE_PROJECT_3, "checkstyle.xml")

SAMPLE_PROJECT_4 = os.path.join(CURR_DIR, "../../res/sample_project_4")
CHECKSTYLE_CONFIG_4 = os.path.join(SAMPLE_PROJECT_4, "checkstyle.xml")

ACTIVITI_PROJECT = os.path.join(CURR_DIR, "../../res/activiti")
CHECKSTYLE_CONFIG_ACTIVITI = os.path.join(ACTIVITI_PROJECT, "checkstyle-rules.xml")


def test_run_checkstyle_on_dir() -> None:
    report = run_checkstyle_on_dir(Path(SAMPLE_PROJECT), "8.0")
    assert len(report) == 4
    for file_report in report:
        if "non" in file_report.path.name or file_report.path.name.endswith("pom.xml"):
            assert len(file_report.violations) == 0
        else:
            assert len(file_report.violations) > 0


def test_run_checkstyle_on_code() -> None:
    report = run_checkstyle_on_str(
        "public class Main() {}", "8.0", Path(CHECKSTYLE_CONFIG)
    )
    assert len(report.violations) == 0


def test_returns_n_violations() -> None:
    returns_n_violation = returns_n_violations(0, "8.0", Path(CHECKSTYLE_CONFIG))
    assert (
        returns_n_violation(lambda: "public class Main() {}")()
        == "public class Main() {}"
    )
    with pytest.raises(WrongViolationAmountException):
        returns_n_violations(42, "8.0", Path(CHECKSTYLE_CONFIG))(
            lambda: "public class Main() {}"
        )()


def _compare_line_by_line(expected: str, actual: str) -> None:
    """
    Compare two files line by line.
    :param expected: The expected file
    :param actual: The actual file
    :return: True if the files are equal, False otherwise
    """
    with open(expected) as expected_file, open(actual) as actual_file:
        # Skip the last line because of newline
        for expected_line, actual_line in zip(
            expected_file.readlines()[:-1], actual_file.readlines()[:-1], strict=True
        ):
            assert expected_line == actual_line


def test_remove_relative_paths() -> None:
    """
    The checkstyle.xml file in sample_project_3 contains relative paths.
    This test checks that the relative paths are removed.
    """
    save = tempfile.mkdtemp()
    fix_checkstyle_config(CHECKSTYLE_CONFIG_3, save / Path("checkstyle-modified.xml"))

    actual_path = os.path.join(save, "checkstyle-modified.xml")
    expected_path = os.path.join(SAMPLE_PROJECT_3, "checkstyle-modified.xml")

    _compare_line_by_line(expected_path, actual_path)


def test_fix_checkstyle() -> None:
    save = tempfile.mkdtemp()
    fix_checkstyle_config(
        CHECKSTYLE_CONFIG_ACTIVITI, save / Path("checkstyle-fixed.xml")
    )

    actual_path = os.path.join(save, "checkstyle-fixed.xml")
    expected_path = os.path.join(ACTIVITI_PROJECT, "checkstyle-fixed.xml")

    _compare_line_by_line(expected_path, actual_path)

    shutil.rmtree(save)
