import os.path
import re
import subprocess
import xml.etree.ElementTree as Xml
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from streamerate import stream

from src.styler2_0.utils.utils import save_content_to_file

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CHECKSTYLE_DIR = os.path.join(CURR_DIR, "../../../checkstyle")
AVAILABLE_VERSIONS = [
    file.split("-")[1]
    for file in os.listdir(CHECKSTYLE_DIR)
    if file.startswith("checkstyle-")
]

# Checkstyle command to execute without test resource files.
CHECKSTYLE_RUN_CMD = (
    "java -jar {} -f xml -c {} {} "
    "--exclude-regexp .*/test/.* "
    "--exclude-regexp .*/resources/.*"
)
CHECKSTYLE_JAR_NAME = "checkstyle-{}-all.jar"
CHECKSTYLE_CONF_REG = re.compile(
    r".?((checkstyle)|(check-style)|(sun)|(google))[-_]?"
    r"((config)|(configuration)|(checks)|(checker)|(rules))?.xml"
)
CHECKSTYLE_TEMP_PATH = Path(os.path.join(CURR_DIR, "../../../checkstyle-tmp"))
JAVA_TEMP_FILE = CHECKSTYLE_TEMP_PATH / Path("Temp.java")

# DOTALL is needed to match multiline comments.
XML_COMMENT_REG = re.compile(r"<!--.*?-->", re.DOTALL)


class NotSuppoertedVersionException(Exception):
    """
    Exception that is raised whenever no suitable checkstyle version is found.
    """


class WrongViolationAmountException(Exception):
    """
    Exception that is raised whenever a code snippet does not contain the expected
    number of violations.
    """


class ViolationType(Enum):
    """
    Supported checkstyle violations.
    """

    ANNOTATION_LOCATION = "AnnotationLocation"
    ANNOTATION_ON_SAME_LINE = "AnnotationOnSameLine"
    COMMENTS_INDENTATION = "CommentsIndentation"
    EMPTY_FOR_INITIALIZER_PAD = "EmptyForInitializerPad"
    EMPTY_FOR_ITERATOR_PAD = "EmptyForIteratorPad"
    EMPTY_LINE_SEPERATOR = "EmptyLineSeparator"
    FILE_TAB_CHARACTER = "FileTabCharacter"
    GENERIC_WHITESPACE = "GenericWhitespace"
    INDENTATION = "Indentation"
    LEFT_CURLY = "LeftCurly"
    LINE_LENGTH = "LineLength"
    METHOD_PARAM_PAD = "MethodParamPad"
    NEW_LINE_AT_END_OF_FILE = "NewlineAtEndOfFile"
    NO_LINE_WRAP = "NoLineWrap"
    NO_WHITESPACE_AFTER = "NoWhitespaceAfter"
    NO_WHITESPACE_BEFORE = "NoWhitespaceBefore"
    ONE_STATEMENT_PER_LINE = "OneStatementPerLine"
    OPERATOR_WRAP = "OperatorWrap"
    PAREN_PAD = "ParenPad"
    REGEXP = "Regexp"
    REGEXP_MULTI_LINE = "RegexpMultiline"
    REGEXP_SINGLE_LINE = "RegexpSingleline"
    REGEXP_SINGLE_LINE_JAVA = "RegexpSinglelineJava"
    RIGHT_CURLY = "RightCurly"
    SEPERATOR_WRAP = "SeparatorWrap"
    SINGLE_SPACE_SEPERATOR = "SingleSpaceSeparator"
    TRAILING_COMMENT = "TrailingComment"
    TYPECAST_PAREN_PAD = "TypecastParenPad"
    WHITESPACE_AFTER = "WhitespaceAfter"
    WHITESPACE_AROUND = "WhitespaceAround"
    NOT_SUPPORTED = "NOT_SUPPORTED"

    @classmethod
    def _missing_(cls, value: object) -> NOT_SUPPORTED:
        return cls.NOT_SUPPORTED


@dataclass(eq=True, frozen=True)
class Violation:
    """
    Class representing a violation and its position.
    """

    type: ViolationType
    line: int
    column: int | None


@dataclass(eq=True, frozen=True)
class CheckstyleFileReport:
    """
    Checkstyle report for a file.
    """

    path: Path | None
    violations: frozenset[Violation]


def find_checkstyle_config(directory: Path) -> Path:
    for subdir, _, files in os.walk(directory):
        for file in files:
            if CHECKSTYLE_CONF_REG.match(file):
                return Path(subdir) / Path(file)
    raise ValueError("Given directory does not contain a checkstyle config!")


def run_checkstyle_on_dir(
    directory: Path, version: str, config: Path = None
) -> frozenset[CheckstyleFileReport]:
    """
    Run checkstyle on the given directory. Returns a set of ChecksStyleFileReport.
    :param config: The given config file.
    :param directory: The directory of the Java project.
    :param version: The version of checkstyle to use.
    :return: Returns a set of ChecksStyleFileReport.
    """
    path_to_jar = _build_path_to_checkstyle_jar(version)
    path_to_checkstyle_config = config if config else find_checkstyle_config(directory)
    checkstyle_cmd = CHECKSTYLE_RUN_CMD.format(
        path_to_jar, path_to_checkstyle_config, directory
    )
    with subprocess.Popen(
        checkstyle_cmd.split(), stdout=subprocess.PIPE
    ) as checkstyle_process:
        output = checkstyle_process.communicate()[0]
        if checkstyle_process.returncode > 0:
            output = b"".join(output.split(b"</checkstyle>")[0:-1]) + b"</checkstyle>"
        return _parse_checkstyle_xml_report(output)


def run_checkstyle_on_str(
    code: str, version: str, config: Path
) -> CheckstyleFileReport:
    """
    Runs checkstyle on the given code snippet.
    :param code: The given snippet.
    :param version: The checkstyle version.
    :param config: The checkstyle config.
    :return: Returns the CheckstyleReport of the snippet.
    """
    os.makedirs(CHECKSTYLE_TEMP_PATH, exist_ok=True)
    save_content_to_file(JAVA_TEMP_FILE, code)
    return list(run_checkstyle_on_dir(CHECKSTYLE_TEMP_PATH, version, config))[0]


def n_violations_in_code(n: int, code: str, version: str, config: Path) -> None:
    """
    Ensures that the given code snippet contains exactly n violations.
    :param n: The given n.
    :param code: The given snippet.
    :param version: Version of checkstyle.
    :param config: Checkstyle config.
    :return:
    """
    report = run_checkstyle_on_str(code, version, config)
    if len(report.violations) != n:
        raise WrongViolationAmountException(
            f"Expected {n} violations got {len(report.violations)}."
        )


def returns_n_violations(
    n: int,
    checkstyle_version: str,
    checkstyle_config: Path | str,
    use_instance: bool = False,
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """
    Decorator ensuring a function decorated with this is returning a string that
    contains exactly the expected amount of violations.

    One can either supply the path to the checkstyle config and the version
    immediately or tell the program to use an instance which holds the values.
    If so the name of the instance variables must be supplied in order to get the
    values.

    :param n: The expected amount of violations.
    :param checkstyle_version: Checkstyle version or variable name of the version.
    :param checkstyle_config: Path to checkstyle config or variable name of the config
                              path.
    :param use_instance: Bool flag to tell the decorator use instance or not.
    :return: Returns the decorated function.
    """

    def _n_violations_decorator(func: Callable[..., str]) -> Callable[..., str]:
        def _returns_n_violations(*args: ..., **kwargs: ...) -> str:
            return_value = func(*args, **kwargs)
            if use_instance:
                self = args[0]
                checkstyle_v = getattr(self, checkstyle_version)
                checkstyle_c = getattr(self, checkstyle_config)
            else:
                checkstyle_v = checkstyle_version
                checkstyle_c = checkstyle_config
            n_violations_in_code(n, return_value, checkstyle_v, checkstyle_c)
            return return_value

        return _returns_n_violations

    return _n_violations_decorator


def contains_config_variables(config: Path) -> bool:
    """
    Checks if the given config contains config variables.
    :param config: The given config.
    :return: Returns true if the config contains config variables.
    """
    config_content = config.read_text()
    return "${" in re.sub(XML_COMMENT_REG, "", config_content)


def _build_path_to_checkstyle_jar(version: str) -> Path:
    return Path(CHECKSTYLE_DIR) / CHECKSTYLE_JAR_NAME.format(version)


def _parse_checkstyle_xml_report(report: bytes) -> frozenset[CheckstyleFileReport]:
    root = Xml.fromstring(report)
    return frozenset(
        stream(list(root))
        .map(
            lambda file: CheckstyleFileReport(
                Path(file.attrib["name"]), _parse_violations(list(file))
            )
        )
        .to_set()
    )


def _parse_violations(raw_violations: list[Xml.Element]) -> frozenset[Violation]:
    return frozenset(
        stream(raw_violations)
        .filter(lambda raw_violation: raw_violation.tag == "error")
        .map(
            lambda raw_violation: Violation(
                ViolationType(_get_violation_name(raw_violation)),
                int(raw_violation.attrib["line"]),
                int(raw_violation.attrib["column"])
                if "column" in raw_violation.attrib
                else None,
            )
        )
        .filter(lambda violation: violation.type != ViolationType.NOT_SUPPORTED)
        .to_set()
    )


def _get_violation_name(raw_violation: Xml.Element) -> str:
    return raw_violation.attrib.get("source").split(".")[-1].replace("Check", "")


def find_version_by_trying(config: Path, project_dir: Path) -> str:
    """
    If there is not a checkstyle version provided, find the correct version
    by trying out all available ones.
    :param config: The path to the found config.
    :param project_dir: The directory
    :return: Returns the working version
    """
    for version in AVAILABLE_VERSIONS:
        try:
            run_checkstyle_on_dir(project_dir, version, config)
            return version
        except Xml.ParseError:
            continue
    raise NotSuppoertedVersionException("No suitable checkstyle version found.")
