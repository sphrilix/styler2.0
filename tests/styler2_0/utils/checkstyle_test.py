import os.path
from pathlib import Path

import pytest

from styler2_0.utils.checkstyle import (
    WrongViolationAmountException,
    returns_n_violations,
    run_checkstyle_on_dir,
    run_checkstyle_on_str,
)

CURR_DIR = os.path.dirname(os.path.relpath(__file__))
SAMPLE_PROJECT = os.path.join(CURR_DIR, "../../res/sample_project")
CHECKSTYLE_CONFIG = os.path.join(SAMPLE_PROJECT, "checkstyle.xml")


def test_run_checkstyle_on_dir() -> None:
    report = run_checkstyle_on_dir(Path(SAMPLE_PROJECT), "8.0")
    assert len(report) == 3
    for file_report in report:
        if "non" in file_report.path.name:
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
