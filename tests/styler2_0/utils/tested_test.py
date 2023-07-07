import os.path
from pathlib import Path

from src.styler2_0.utils.tested import extract_tested_src_files

CURR_DIR = os.path.dirname(os.path.relpath(__file__))
SAMPLE_PROJECT_5 = os.path.join(CURR_DIR, "../../res/sample_project_5")


def test_split_test_files() -> None:
    # load all files in SAMPLE_PROJECT_5 as set of paths
    input_files = set()
    for root, _, files in os.walk(SAMPLE_PROJECT_5):
        for file in files:
            input_files.add(Path(os.path.join(root, file)))

    tested_files = extract_tested_src_files(input_files)
    assert len(tested_files) == 1
