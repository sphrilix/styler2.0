import os.path
import re
import subprocess
import xml.etree.ElementTree as Xml
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory

from streamerate import stream

from src.styler2_0.utils.utils import read_content_of_file, save_content_to_file

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
    r"((config)|(configuration)|(checks)|(checker)|(rules)|(litterbox))?.xml"
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
    # NEW_LINE_AT_END_OF_FILE = "NewlineAtEndOfFile"
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
    ABBREVIATION_AS_WORD_IN_NAME = "AbbreviationAsWordInName"
    ABSTRACT_CLASS_NAME = "AbstractClassName"
    CATCH_PARAMETER_NAME = "CatchParameterName"
    CLASS_TYPE_PARAMETER_NAME = "ClassTypeParameterName"
    CONSTANT_NAME = "ConstantName"
    ILLEGAL_IDENTIFIER_NAME = "IllegalIdentifierName"
    INTERFACE_TYPE_PARAMETER_NAME = "InterfaceTypeParameterName"
    LAMBDA_PARAMETER_NAME = "LambdaParameterName"
    LOCAL_FINAL_VARIABLE_NAME = "LocalFinalVariableName"
    LOCAL_VARIABLE_NAME = "LocalVariableName"
    MEMBER_NAME = "MemberName"
    METHOD_NAME = "MethodName"
    METHOD_TYPE_PARAMETER_NAME = "MethodTypeParameterName"
    PACKAGE_NAME = "PackageName"
    PARAMETER_NAME = "ParameterName"
    PATTERN_VARIABLE_NAME = "PatternVariableName"
    RECORD_COMPONENT_NAME = "RecordComponentName"
    RECORD_TYPE_PARAMETER_NAME = "RecordTypeParameterName"
    STATIC_VARIABLE_NAME = "StaticVariableName"
    TYPE_NAME = "TypeName"

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

        # Remove trailing non xml content. Apparently, it seems
        # to fix the ParserError.
        # IDK why styler does this if and only if checkstyle
        # terminates with an error.
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
    return next(iter(run_checkstyle_on_strs({0: code}, version, config).values()))


def run_checkstyle_on_strs(
    codes: dict[int, str], version: str, config: Path
) -> dict[int, CheckstyleFileReport]:
    """
    Runs checkstyle on the given code snippets all at once.
    Reports back the reports and their id.
    The codes are a dict with an id to later match reports to corresponding strs.
    The report.path == {temp_dir}/id.java.
    :param codes: Snippets with given identifier.
    :param version: Checkstyle version to be used.
    :param config: Checkstyle config to be used.
    :return: Returns a dictionary with the id and the corresponding report.
    """
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        for id, code in codes.items():
            save_content_to_file(temp_dir / f"{id}.java", code)
        reports = run_checkstyle_on_dir(temp_dir, version, config)
        report_dict: dict[int, CheckstyleFileReport] = {}
        for report in reports:
            id = int(report.path.name.replace(".java", ""))
            report_dict[id] = report
    return report_dict


def n_violations_in_code(n: int, code: str, version: str, config: Path) -> None:
    """
    Ensures that the given code snippet contains exactly n violations.
    :param n: The given n.
    :param code: The given snippet.
    :param version: Version of checkstyle.
    :param config: Checkstyle config.
    :return:
    """
    # TODO: Why sometimes parse error?
    report = run_checkstyle_on_str(code, version, config)
    if not report or len(report.violations) != n:
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
    config_content = read_content_of_file(config)
    return "${" in re.sub(XML_COMMENT_REG, "", config_content)


def _build_path_to_checkstyle_jar(version: str) -> Path:
    return Path(CHECKSTYLE_DIR) / CHECKSTYLE_JAR_NAME.format(version)


def _parse_checkstyle_xml_report(report: bytes) -> frozenset[CheckstyleFileReport]:
    root = Xml.fromstring(report)
    return frozenset(
        stream(list(root))
        .filter(lambda file: file.attrib["name"].endswith(".java"))
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
