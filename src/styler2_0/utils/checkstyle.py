import os.path
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from streamerate import stream

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


class ViolationType(Enum):
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
    type: ViolationType
    line: int
    column: int


@dataclass(eq=True, frozen=True)
class CheckstyleFileReport:
    path: Path
    violations: frozenset[Violation]


def _find_checkstyle_config(directory: Path) -> Path:
    for subdir, _, files in os.walk(directory):
        for file in files:
            if CHECKSTYLE_CONF_REG.match(file):
                return Path(subdir) / Path(file)
    raise ValueError("Given directory does not contain a checkstyle config!")


def run_checkstyle_on_dir(
    directory: Path, version: str
) -> frozenset[CheckstyleFileReport]:
    """
    Run checkstyle on the given directory. Returns a set of ChecksStyleFileReport.
    :param directory: The directory of the Java project.
    :param version: The version of checkstyle to use.
    :return: Returns a set of ChecksStyleFileReport.
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
        if checkstyle_process.returncode > 0:
            output = b"".join(output.split(b"</checkstyle>")[0:-1]) + b"</checkstyle>"
        return _parse_checkstyle_xml_report(output)


def _build_path_to_checkstyle_jar(version: str) -> Path:
    return Path(CHECKSTYLE_DIR) / CHECKSTYLE_JAR_NAME.format(version)


def _parse_checkstyle_xml_report(report: bytes) -> frozenset[CheckstyleFileReport]:
    root = ET.fromstring(report)
    return frozenset(
        stream(list(root))
        .map(
            lambda file: CheckstyleFileReport(
                Path(file.attrib["name"]), _parse_violations(list(file))
            )
        )
        .to_set()
    )


def _parse_violations(raw_violations: list[ET.Element]) -> frozenset[Violation]:
    return frozenset(
        stream(raw_violations)
        .filter(lambda raw_violation: raw_violation.tag == "error")
        .map(
            lambda raw_violation: Violation(
                ViolationType(_get_violation_name(raw_violation)),
                int(raw_violation.attrib["line"]),
                int(raw_violation.attrib.get("column", 0)),
            )
        )
        .filter(lambda violation: violation.type != ViolationType.NOT_SUPPORTED)
        .to_set()
    )


def _get_violation_name(raw_violation: ET.Element) -> str:
    return raw_violation.attrib.get("source").split(".")[-1].replace("Check", "")
