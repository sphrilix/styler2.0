import shutil
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree.ElementTree import ParseError

from pydriller import Commit, Git, Repository
from streamerate import stream

from src.styler2_0.utils.checkstyle import (
    CheckstyleFileReport,
    ViolationType,
    run_checkstyle_on_dir,
)

MINED_VIOLATIONS_DIR = Path("mined_violations")

# TODO: How to handle suppression files?!
#       Styler for example discards every commit that contains a suppression file.


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
    violation_report: CheckstyleFileReport
    violations_hash: str
    fix_report: CheckstyleFileReport | None = field(default=None, init=False, repr=True)
    fix_hash: str | None = field(default=None, init=False, repr=True)

    def is_fixed(self) -> bool:
        return self.fix_report is not None


def process_git_repository(
    input_dir: Path, output_dir: Path, version: str, config: Path
) -> None:
    with GitContextManager(input_dir) as git_repo:
        commits = _extract_commits_to_search(input_dir, config)
        violations = _mine_violations_and_fixes_from_commits(
            commits, config, version, input_dir, git_repo
        )
        _save_violations(violations, input_dir, output_dir)


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
    for commit in commits:
        # TODO: check why the parse error is raised
        with suppress(ParseError):
            git_repo.checkout(commit.hash)
            checkstyle_reports = run_checkstyle_on_dir(input_dir, version, config)
            interesting_reports_of_commit = _filter_interesting_reports(
                checkstyle_reports
            )

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
                [
                    MinedViolation(report, commit.hash)
                    for report in new_reports_of_commit
                ]
            )
    return list(mined_violations)


def _update_violation_with_fix(
    non_affected_files: set[CheckstyleFileReport],
    not_fixed_violations: list[MinedViolation],
    commit_hash: str,
) -> None:
    for violation in not_fixed_violations:
        for non_affected_file in non_affected_files:
            if violation.violation_report.path == non_affected_file.path:
                violation.fix_report = non_affected_file
                violation.fix_hash = commit_hash


def _get_new_reports(
    all_reports: set[CheckstyleFileReport],
    to_be_checked_reports: frozenset[CheckstyleFileReport],
) -> frozenset[CheckstyleFileReport]:
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
    return str(config.name) in (
        modified_file.filename for modified_file in commit.modified_files
    )


def _save_violations(
    commit_reports: list[MinedViolation], input_dir: Path, output_dir: Path
) -> None:
    git_repo = Git(str(input_dir))
    for i, violation in enumerate(commit_reports):
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
