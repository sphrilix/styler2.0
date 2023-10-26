import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydriller import Commit, Git, Repository
from streamerate import stream
from tqdm import tqdm

from src.styler2_0.utils.checkstyle import (
    CheckstyleFileReport,
    ViolationType,
    contains_config_variables,
    run_checkstyle_on_dir,
)
from src.styler2_0.utils.maven import pom_includes_checkstyle_suppression
from src.styler2_0.utils.utils import save_content_to_file

MINED_VIOLATIONS_DIR = Path("mined_violations")

# TODO: How to handle suppression files?!
#       Styler for example discards every commit that contains a suppression file.


class NotProcessableGitRepositoryException(Exception):
    """
    Exception that is raised whenever a git repository is not processable.
    """


class GitContextManager:
    """
    Context manager that ensures that the git repository is in the same state
    as before the context manager was entered. Actually, ensuring that the
    last commit is checked out.
    """

    def __init__(self, input_dir: Path) -> None:
        self._git = Git(str(input_dir))
        self._current_commit = self._git.get_head()

    def __enter__(self) -> Git:
        self._git.checkout(self._current_commit.hash)
        return self._git

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: type) -> None:
        self._git.checkout(self._current_commit.hash)


@dataclass(eq=True)
class MinedViolation:
    """
    Dataclass that represents a mined violation.
    """

    violation_report: CheckstyleFileReport
    violations_hash: str
    fix_report: CheckstyleFileReport | None = field(default=None, init=False, repr=True)
    fix_hash: str | None = field(default=None, init=False, repr=True)

    def is_fixed(self) -> bool:
        """
        Returns whether the violation is fixed or not.
        :return: True if the violation is fixed, False otherwise.
        """
        return self.fix_report is not None


def process_git_repository(
    input_dir: Path, output_dir: Path, version: str, config: Path
) -> None:
    """
    Processes the git repository in the input directory and saves the mined violations
    :param input_dir: The input directory of the git repository.
    :param output_dir: The output directory where the mined violations are saved.
    :param version: The checkstyle version that is used.
    :param config: The checkstyle config file that is used.
    """
    if contains_config_variables(config):
        raise NotProcessableGitRepositoryException(
            f"Config file {config} contains variables!"
        )
    with GitContextManager(input_dir) as git_repo:
        commits = _extract_commits_to_search(input_dir, config)
        violations = _mine_violations_and_fixes_from_commits(
            commits, config, version, input_dir, git_repo
        )
        _save_violations(violations, input_dir, output_dir)
        meta_data = {
            "version": version,
            "config": str(config),
        }
        save_content_to_file(
            output_dir / MINED_VIOLATIONS_DIR / "data.json", json.dumps(meta_data)
        )


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


def _mine_violations_and_fixes_from_commits(
    commits: Sequence[Commit],
    config: Path,
    version: str,
    input_dir: Path,
    git_repo: Git,
) -> list[MinedViolation]:
    """
    Mines the violations from the commits.
    :param commits: commits to mine violations from
    :param config: path of checkstyle config file
    :param version: checkstyle version
    :return: Returns the mined violations.
    """

    already_seen_reports: set[CheckstyleFileReport] = set()
    mined_violations: list[MinedViolation] = []
    for commit in tqdm(commits, desc="Mining violations"):
        # TODO: check why the parse error is raised
        git_repo.checkout(commit.hash)
        if pom_includes_checkstyle_suppression(input_dir):
            continue
        checkstyle_reports = run_checkstyle_on_dir(input_dir, version, config)
        interesting_reports_of_commit = _filter_interesting_reports(checkstyle_reports)

        # Get new reports of commit as the same violation can occur in
        # different commits
        new_reports_of_commit = _get_new_reports(
            already_seen_reports, interesting_reports_of_commit
        )

        # Get the files with no violations
        non_affected_files = {
            report for report in checkstyle_reports if not report.violations
        }

        violations_with_no_fix = _violations_with_no_fix(mined_violations)

        # Update the violations with no fix, if the violation is fixed in
        # the current commit which means the file is now in the non_affected_files
        _update_violation_with_fix(
            non_affected_files, violations_with_no_fix, commit.hash
        )

        already_seen_reports.update(set(new_reports_of_commit))

        # Update mined_violations with the new violations
        mined_violations.extend(
            [MinedViolation(report, commit.hash) for report in new_reports_of_commit]
        )
    return list(mined_violations)


def _update_violation_with_fix(
    non_affected_files: set[CheckstyleFileReport],
    not_fixed_violations: list[MinedViolation],
    commit_hash: str,
) -> None:
    """
    Updates the violations with no fix, if the violation is fixed in the current commit
    :param non_affected_files: The files with no violations
    :param not_fixed_violations: The violations with no fix
    :param commit_hash: The hash of the commit
    :return:
    """
    for violation in not_fixed_violations:
        for non_affected_file in non_affected_files:
            if violation.violation_report.path == non_affected_file.path:
                violation.fix_report = non_affected_file
                violation.fix_hash = commit_hash


def _get_new_reports(
    all_reports: set[CheckstyleFileReport],
    to_be_checked_reports: frozenset[CheckstyleFileReport],
) -> frozenset[CheckstyleFileReport]:
    """
    Returns the reports that are not already seen.
    :param all_reports: All reports that are already seen.
    :param to_be_checked_reports: Possible new reports.
    :return: New reports.
    """
    return frozenset(
        report for report in to_be_checked_reports if report not in all_reports
    )


def _violations_with_no_fix(violations: list[MinedViolation]) -> list[MinedViolation]:
    """
    Returns the violations that have no fix.
    :param violations: All violations
    :return:
    """
    return list(
        stream(violations).filter(lambda violation: not violation.is_fixed()).to_list()
    )


def _filter_interesting_reports(
    reports: frozenset[CheckstyleFileReport],
) -> frozenset[CheckstyleFileReport]:
    """
    Filters the reports that containS exactly one or none violations.
    :param reports: checkstyle reports to be filtered.
    :return: filtered reports.
    """
    return frozenset(
        stream(list(reports))
        .filter(
            lambda report: len(report.violations) == 1
            and next(iter(report.violations)).type != ViolationType.NOT_SUPPORTED
        )
        .to_set()
    )


def _is_config_modified(commit: Commit, config: Path) -> bool:
    """
    Checks if the checkstyle config file is modified in the commit.
    :param commit: The commit to be checked.
    :param config: The Path of the checkstyle config file.
    :return: True if the config file is modified, False otherwise.
    """
    return str(config.name) in (
        modified_file.filename for modified_file in commit.modified_files
    )


def _save_violations(
    commit_reports: list[MinedViolation], input_dir: Path, output_dir: Path
) -> None:
    """
    Saves the violations and fixes in the output_dir.
    :param commit_reports: The mined violations.
    :param input_dir: The input dir.
    :param output_dir: The output dir.
    :return:
    """
    git_repo = Git(str(input_dir))
    for i, violation in tqdm(enumerate(commit_reports), desc="Saving violations"):
        git_repo.checkout(violation.violations_hash)
        violation_dir = output_dir / MINED_VIOLATIONS_DIR / str(i) / "violation/"
        violation_dir.mkdir(parents=True)
        shutil.copy(
            violation.violation_report.path,
            violation_dir,
        )
        if violation.is_fixed():
            fix_dir = output_dir / MINED_VIOLATIONS_DIR / str(i) / "fix/"
            fix_dir.mkdir(parents=True)
            git_repo.checkout(violation.fix_hash)
            shutil.copy(
                violation.fix_report.path,
                fix_dir,
            )
