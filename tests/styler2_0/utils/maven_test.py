import os
from pathlib import Path

from src.styler2_0.utils.maven import get_checkstyle_version_of_project

CURR_DIR = os.path.dirname(os.path.relpath(__file__))
SAMPLE_PROJECT = os.path.join(CURR_DIR, "../../res/sample_project")
SAMPLE_PROJECT_2 = os.path.join(CURR_DIR, "../../res/sample_project_2")


def test_get_checkstyle_version_of_sample_project() -> None:
    version = get_checkstyle_version_of_project(Path(SAMPLE_PROJECT))
    assert version == "8.29.0"


def test_get_checkstyle_with_specified_runtime_version() -> None:
    version = get_checkstyle_version_of_project(Path(SAMPLE_PROJECT_2))
    assert version == "9.3.0"
