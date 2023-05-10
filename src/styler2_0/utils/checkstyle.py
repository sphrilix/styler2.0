import os.path
import re
import subprocess
from pathlib import Path

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CHECKSTYLE_DIR = os.path.join(CURR_DIR, "../../../checkstyle")
AVAILABLE_VERSIONS = [
    file.split("-")[1]
    for file in os.listdir(CHECKSTYLE_DIR)
    if file.startswith("checkstyle-")
]
CHECKSTYLE_RUN_CMD = (
    "java -jar {} -f xml -c {} {} "
    "--exclude-regexp .*/test/.* "
    "--exclude-regexp .*/resources/.*"
)
CHECKSTYLE_JAR_NAME = "checkstyle-{}-all.jar"
CHECKSTYLE_CONF_REG = re.compile(r".*checkstyle.*\.xml")


def _find_checkstyle_config(directory: Path) -> Path:
    for subdir, _, files in os.walk(directory):
        for file in files:
            if CHECKSTYLE_CONF_REG.match(file):
                return Path(subdir) / Path(file)
    raise ValueError("Given directory does not contain a checkstyle config!")


def run_checkstyle_on_dir(directory: Path, version: str) -> None:
    """
    Run checkstyle on the given directory.
    :param directory:
    :param version:
    :return:
    """
    path_to_jar = _build_path_to_checkstyle_jar(version)
    path_to_checkstyle_config = _find_checkstyle_config(directory)
    checkstyle_cmd = CHECKSTYLE_RUN_CMD.format(
        path_to_jar, path_to_checkstyle_config, directory
    )
    with subprocess.Popen(
        checkstyle_cmd.split(), stdout=subprocess.PIPE
    ) as checkstyle_process:
        output = checkstyle_process.communicate()[0]
    print(output)


def _build_path_to_checkstyle_jar(version: str) -> Path:
    return Path(CHECKSTYLE_DIR) / CHECKSTYLE_JAR_NAME.format(version)
