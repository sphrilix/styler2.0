import os.path
import shutil
import tempfile
from pathlib import Path

import pytest

from src.styler2_0.utils.checkstyle import (
    WrongViolationAmountException,
    _remove_relative_paths,
    returns_n_violations,
    run_checkstyle_on_dir,
    run_checkstyle_on_str,
)
from styler2_0.utils.utils import read_content_of_file

CURR_DIR = os.path.dirname(os.path.relpath(__file__))
SAMPLE_PROJECT = os.path.join(CURR_DIR, "../../res/sample_project")
CHECKSTYLE_CONFIG = os.path.join(SAMPLE_PROJECT, "checkstyle.xml")

RELATIVE_PATHS_PROJECT = os.path.join(
    CURR_DIR, "../../res/sample_project_relative_paths"
)
CHECKSTYLE_RELATIVE_PATHS = os.path.join(RELATIVE_PATHS_PROJECT, "checkstyle.xml")


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


def test_remove_relative_paths() -> None:
    """
    The checkstyle.xml file in sample_project_3 contains relative paths.
    This test checks that the relative paths are removed.
    """
    save = tempfile.mkdtemp()
    _remove_relative_paths(
        Path(CHECKSTYLE_RELATIVE_PATHS), save / Path("checkstyle-modified.xml")
    )

    actual_path = Path(os.path.join(save, "checkstyle-modified.xml"))
    expected_path = Path(
        os.path.join(RELATIVE_PATHS_PROJECT, "checkstyle-modified.xml")
    )

    # compare expected and actual file
    expected_content = read_content_of_file(expected_path)
    actual_content = read_content_of_file(actual_path)
    for expected_line, actual_line in zip(
        expected_content.splitlines(), actual_content.splitlines(), strict=True
    ):
        assert expected_line == actual_line

    shutil.rmtree(save)
