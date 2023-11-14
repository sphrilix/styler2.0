import os
import re
import xml.etree.ElementTree as Xml
from pathlib import Path

from styler2_0.utils.utils import read_content_of_file

MAVEN_PLUGIN_CHECKSTYLE_VERSION = {
    "3.3.0": "9.3",
    "3.2.2": "9.3",
    "3.2.1": "9.3",
    "3.2.0": "9.3",
    "3.1.2": "8.29",
    "3.1.1": "8.29",
    "3.1.0": "8.19",
    "3.0.0": "8.18",
}
POM_XML = "pom.xml"
STANDARD_NAMESPACE = "http://maven.apache.org/POM/4.0.0"
CHECKSTYLE_PLUGIN_ARTIFACT_ID = "maven-checkstyle-plugin"
CHECKSTYLE_ARTIFACT_ID = "checkstyle"
DEPENDENCY_REGEX = re.compile(r"\${[^}]+}")
SUPPRESSION_LOCATION = "suppressionsLocation"


class MavenException(Exception):
    """
    Exception thrown whenever something wrong with pom.xml.
    """


def _find_root_pom_xml(project_dir: Path) -> Path:
    pom_files = [
        Path(os.path.join(subdir, POM_XML))
        for subdir, _, files in os.walk(project_dir)
        if POM_XML in files
    ]
    sorted_pom_files = sorted(pom_files, key=lambda x: len(x.parts))
    if not sorted_pom_files:
        raise MavenException(f"No {POM_XML} found in {project_dir}.")
    return sorted_pom_files[0]


def _get_checkstyle_version_from_pom(pom: Path) -> str:
    if not str(pom).endswith(POM_XML):
        raise ValueError(f"No {POM_XML} was given")
    parsed_pom = Xml.parse(pom)
    root = parsed_pom.getroot()
    namespaces = _get_namespaces(root)
    for plugin in root.findall(".//xmlns:plugin", namespaces=namespaces):
        name = plugin.find("xmlns:artifactId", namespaces=namespaces).text
        if name != CHECKSTYLE_PLUGIN_ARTIFACT_ID:
            continue
        return _parse_checkstyle_version_from_xml_elem(plugin, root, namespaces)
    raise MavenException("Project does not support checkstyle.")


def _get_namespaces(root: Xml.Element) -> dict[str, str]:
    namespaces = re.match(r"{(.*)}", root.tag)
    return {"xmlns": namespaces.group(1) if namespaces else STANDARD_NAMESPACE}


def _parse_checkstyle_version_from_xml_elem(
    elem: Xml.Element, root: Xml.Element, namespaces: dict[str, str]
) -> str:
    checkstyle_dependencies = elem.findall(".//xmlns:dependency", namespaces=namespaces)
    for dependency in checkstyle_dependencies:
        name = dependency.find("xmlns:artifactId", namespaces=namespaces).text
        if name == CHECKSTYLE_ARTIFACT_ID:
            version = dependency.find("xmlns:version", namespaces=namespaces).text
            if version:
                return version
    plugin_version = elem.find("xmlns:version", namespaces=namespaces).text
    if plugin_version:
        if DEPENDENCY_REGEX.match(plugin_version):
            plugin_version = _parse_checkstyle_version_from_variable(
                plugin_version, root, namespaces
            )
        if plugin_version in MAVEN_PLUGIN_CHECKSTYLE_VERSION:
            return MAVEN_PLUGIN_CHECKSTYLE_VERSION[plugin_version]

    raise MavenException("Not supported checkstyle version.")


def get_checkstyle_version_of_project(project_dir: Path) -> str:
    """
    Returns the checkstyle version specified in pom.
    :param project_dir: Directory of the project.
    :return: Return the checkstyle version.
    """
    pom = _find_root_pom_xml(project_dir)
    return _get_checkstyle_version_from_pom(pom)


def pom_includes_checkstyle_suppression(project_dir: Path) -> bool:
    """
    Returns whether the pom includes checkstyle suppressions.
    :param project_dir: Directory of the project.
    :return: Return whether the pom includes checkstyle suppressions.
    """
    pom = _find_root_pom_xml(project_dir)
    pom_content = read_content_of_file(pom)
    return SUPPRESSION_LOCATION in pom_content


def _parse_checkstyle_version_from_variable(
    variable: str, root: Xml.Element, namespaces: dict[str, str]
) -> str | None:
    """
    Parse checkstyle version from variable.
    :param variable: Variable to parse.
    :param root: Root of pom.
    :param namespaces: Namespaces of pom.
    :return: Return the checkstyle version. If not found, return None.
    """
    variable = variable[2:-1]
    for prop in root.findall(".//xmlns:properties", namespaces=namespaces):
        for child in prop:
            if child.tag == f"{{{STANDARD_NAMESPACE}}}{variable}":
                return child.text

    return None
