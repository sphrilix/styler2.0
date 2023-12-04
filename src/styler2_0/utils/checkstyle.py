import logging
import os.path
import re
import subprocess
import xml.etree.ElementTree as Xml
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from shutil import copyfile

from streamerate import stream

from src.styler2_0.utils.maven import DEPENDENCY_REGEX
from src.styler2_0.utils.utils import save_content_to_file

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
CHECKSTYLE_CONF_REG_1 = re.compile(
    r".?((checkstyle)|(check-style)|(sun)|(google))[-_]?"
    r"((config)|(configuration)|(checks)|(checker)|(rules))?.xml"
)
CHECKSTYLE_CONF_REG_2 = re.compile(
    r".*((checkstyle)|(check)|(style)|(sun)|(google)"
    r"|(configuration)|(checks)|(checker)|(rules)).*\.xml"
)
CHECKSTYLE_TEMP_PATH = Path(os.path.join(CURR_DIR, "../../../checkstyle-tmp"))
JAVA_TEMP_FILE = CHECKSTYLE_TEMP_PATH / Path("Temp.java")
CHECKSTYLE_TIMEOUT = 5 * 60  # 5 minutes


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
class CheckstyleReport:
    """
    Checkstyle report for a file.
    """

    path: Path | None
    violations: frozenset[Violation]


def find_checkstyle_config(directory: Path) -> Path:
    """
    Find the checkstyle config in the given directory.
    :param directory: The directory to search in.
    :return: The path to the checkstyle config.
    """
    # CHECKSTYLE_CONF_REG_1 is more specific than CHECKSTYLE_CONF_REG_2
    matches_1 = _find_checkstyle_config(directory, CHECKSTYLE_CONF_REG_1)
    if len(matches_1) == 1:
        return Path(matches_1[0])

    # If no or multiple matches are found, try the other regex and print all results
    matches_2 = _find_checkstyle_config(directory, CHECKSTYLE_CONF_REG_2)

    # If no matches are found there is no checkstyle config
    if len(matches_2) == 0:
        raise ValueError("Given directory does not contain a checkstyle config!")

    # Log all possible configurations using logging
    shortest_match = min(matches_2, key=len)
    if len(matches_2) > 1:
        logging.warning(
            "Found multiple possible checkstyle configs: %s. \n"
            "Using the shortest one: %s",
            ", ".join(matches_2),
            shortest_match,
        )

    return Path(shortest_match)


def _find_checkstyle_config(directory: Path, regex: re.Pattern) -> list[str]:
    """
    Find the checkstyle config in the given directory using the given regex.
    :param directory: The directory to search in.
    :param regex: The regex to use.
    :return: The path to the checkstyle config.
    """
    matches = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if regex.match(file):
                matches.append(os.path.join(subdir, file))
    return matches


def run_checkstyle_on_dir(
    directory: Path,
    version: str,
    config: Path = None,
    timeout: int = CHECKSTYLE_TIMEOUT,
) -> frozenset[CheckstyleReport]:
    """
    Run checkstyle on the given directory. Returns a set of ChecksStyleFileReport.
    :param config: The given config file.
    :param directory: The directory of the Java project.
    :param version: The version of checkstyle to use.
    :param timeout: The timeout for the checkstyle process.
    :return: Returns a set of ChecksStyleFileReport.
    """
    # Build the command to run checkstyle
    path_to_jar = _build_path_to_checkstyle_jar(version)
    path_to_checkstyle_config = config if config else find_checkstyle_config(directory)
    checkstyle_cmd = CHECKSTYLE_RUN_CMD.format(
        path_to_jar, path_to_checkstyle_config, directory
    )

    # Run checkstyle and parse the output
    logging.info("Running checkstyle with command: %s", checkstyle_cmd)
    with subprocess.Popen(
        checkstyle_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    ) as checkstyle_process:
        try:
            report, stderr = checkstyle_process.communicate(timeout=timeout)

            # Log errors of the subprocess
            if stderr:
                logging.error(f"Checkstyle error:\n{stderr}")

            if checkstyle_process.returncode == 4294967294:
                logging.error(
                    "Checkstyle failed to run. Please check if the provided "
                    "project is a maven java project using checkstyle."
                )
            else:
                logging.info(
                    "Checkstyle completed with %s violations",
                    checkstyle_process.returncode,
                )

                # TODO: Why are we not using the xml output but the bytes?
                # Convert the xml to bytes
                report = report.encode("utf-8")
                if checkstyle_process.returncode > 0:
                    report = (
                        b"".join(report.split(b"</checkstyle>")[0:-1])
                        + b"</checkstyle>"
                    )

        # If the subprocess exceeds the timeout, terminate it
        except subprocess.TimeoutExpired:
            checkstyle_process.terminate()
            logging.error("Checkstyle timed out after %s seconds", timeout)
            raise TimeoutError(
                "Checkstyle timed out after %s seconds" % timeout
            ) from None

        return _parse_checkstyle_xml_report(report)


def run_checkstyle_on_str(code: str, version: str, config: Path) -> CheckstyleReport:
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


def _build_path_to_checkstyle_jar(version: str) -> Path:
    return Path(CHECKSTYLE_DIR) / CHECKSTYLE_JAR_NAME.format(version)


def _parse_checkstyle_xml_report(report: bytes) -> frozenset[CheckstyleReport]:
    root = Xml.fromstring(report)
    return frozenset(
        stream(list(root))
        .map(
            lambda file: CheckstyleReport(
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


def fix_checkstyle_config(
    config_path: Path, save_path: Path, version: str = "9.3"
) -> None:
    """
    Fixes the given checkstyle config file by removing all relative paths
    and by removing LineLength under TreeWalker.
    :param config_path: The path to the checkstyle config file.
    :param save_path: The path to save the modified checkstyle config file.
    :param version: The checkstyle version.
    """
    # Create a new XML file with the same name as the original in the save path
    copied_file = copyfile(config_path, save_path)

    # Load the XML file
    tree = Xml.parse(config_path)
    root = tree.getroot()

    # Add parent info to all elements
    _add_parent_info(root)

    # Fix the checkstyle config
    _remove_relative_paths(root)
    _remove_variable_paths(root)

    if version == "9.3":
        _remove_from_modules(root, "TreeWalker", "LineLength")
        _remove_from_modules(root, "TreeWalker", "JavadocMethod", "allowMissingJavadoc")
        _remove_from_modules(root, "TreeWalker", "JavadocMethod", "scope")
        _remove_from_modules(
            root, "TreeWalker", "JavadocMethod", "allowMissingThrowsTags"
        )
        _remove_from_modules(
            root, "TreeWalker", "JavadocMethod", "allowThrowsTagsForSubclasses"
        )
        _remove_from_modules(root, "TreeWalker", "JavadocMethod", "minLineCount")

    if version == "8.29":
        _remove_from_modules(
            root, "TreeWalker", "JavadocMethod", "allowThrowsTagsForSubclasses"
        )
        _remove_from_modules(root, "TreeWalker", "JavadocMethod", "minLineCount")

    # Remove parent info
    _strip_parent_info(root)

    # Write everything until "<module" to the new file
    with open(copied_file, "w") as new_config, open(config_path) as old_config:
        for line in old_config:
            if line.strip().startswith("<module"):
                break
            new_config.write(line)

    # Append the root and all children to the new file
    with open(copied_file, "a") as new_config:
        new_config.write(Xml.tostring(root, encoding="unicode"))


def _remove_relative_paths(root: Xml.Element) -> None:
    """
    Removes all relative paths (starting with "/" followed by a letter) from the
    given XML tree.
    :param root:  The root of the XML tree.
    :return:    Returns the root of the XML tree without relative paths.
    """
    for property_element in root.findall(".//property"):
        if property_element is not None and re.match(
            r"/[A-Za-z]", property_element.get("value")
        ):
            # Remove the element and all children/parents from the tree root
            while property_element not in list(root):
                property_element = _get_parent(property_element)
            root.remove(property_element)


def _remove_variable_paths(root: Xml.Element) -> None:
    """
    Removes all paths that match the dependency regex.
    :param root: The root of the XML tree.
    :return: Returns the root of the XML tree without variable paths.
    """
    for property_element in root.findall(".//property"):
        if (
            property_element is not None
            and property_element.get("value") is not None
            and DEPENDENCY_REGEX.match(property_element.get("value"))
        ):
            _remove_element(property_element)


def _find_modules(root: Xml.Element, parent: str, module: str) -> list[Xml.Element]:
    """
    Finds all modules with the given parent and module name.
    :param root: The root of the XML tree.
    :param parent: The parent of the module.
    :param module: The name of the module.
    :return: A list of all modules with the given parent and module name.
    """
    modules = []
    for module_element in root.findall(".//module"):
        if (
            module_element is not None
            and module_element.get("name") == module
            and _get_parent(module_element).get("name") == parent
        ):
            modules.append(module_element)
    return modules


def _find_properties(root: Xml.Element, module: str, prop: str) -> list[Xml.Element]:
    """
    Finds all properties with the given module and property name.
    :param root: The root of the XML tree.
    :param module: The module of the property.
    :param prop: The name of the property.
    :return: A list of all properties with the given module and property name.
    """
    properties = []
    for property_element in root.findall(".//property"):
        if (
            property_element is not None
            and property_element.get("name") == prop
            and _get_parent(property_element).get("name") == module
        ):
            properties.append(property_element)
    return properties


def _remove_from_modules(root: Xml.Element, *args) -> None:
    """
    Removes the property or module from the given XML tree.
    :param root: The root of the XML tree.
    :param args: The path to the property or module.
    :return: Returns the root of the XML tree without the property or module.
    """
    if len(args) == 2:  # Remove a module
        for module_element in _find_modules(root, args[0], args[1]):
            _remove_element(module_element)

    elif len(args) == 3:  # Remove a property
        for module_element in _find_modules(root, args[0], args[1]):
            for property_element in _find_properties(module_element, args[1], args[2]):
                _remove_element(property_element)

    else:
        raise ValueError("Invalid number of arguments.")


def _remove_element(module_element: Xml.Element) -> None:
    """
    Removes the property with the given name from the given XML tree.
    :param root: The root of the XML tree.
    :param module_element: The module element to remove.
    :return:  Returns the root of the XML tree without the property.
    """
    parent = _get_parent(module_element)
    parent.remove(module_element)


def _add_parent_info(et):
    for child in et:
        child.attrib["__my_parent__"] = et
        _add_parent_info(child)


def _strip_parent_info(et):
    for child in et:
        child.attrib.pop("__my_parent__", "None")
        _strip_parent_info(child)


def _get_parent(et):
    if "__my_parent__" in et.attrib:
        return et.attrib["__my_parent__"]
    return None
