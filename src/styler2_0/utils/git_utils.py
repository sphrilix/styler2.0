from collections.abc import Sequence
from pathlib import Path

from pydriller import Commit, Repository
from streamerate import stream

MINE_VIOLATIONS_DIR = Path("mined_violations")


def process_git_repository(
    input_dir: Path, output_dir: Path, version: str, config: Path
) -> None:
    _extract_commits_to_search(input_dir, config)


def _extract_commits_to_search(input_dir: Path, config: Path) -> Sequence[Commit]:
    """
    Extracts the commits that do not modify the checkstyle config file.
    :param input_dir: input directory of the git repository
    :param config: path of checkstyle config file
    :return: Returns the commits until the last change of the checkstyle config file.
    """
    return list(
        stream(Repository(str(input_dir)).traverse_commits())
        .reversed()
        .takeWhile(lambda commit: not _is_config_modified(commit, config))
        .reversed()
        .to_list()
    )


def _mine_violations_from_commits(
    commits: Sequence[Commit], config: Path, version: str
) -> None:
    """
    Mines the violations from the commits.
    :param commits: commits to mine violations from
    :param config: path of checkstyle config file
    :param version: checkstyle version
    :return: None
    """
    pass


def _is_config_modified(commit: Commit, config: Path) -> bool:
    return str(config.name) in (
        modified_file.filename for modified_file in commit.modified_files
    )
