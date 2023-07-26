import os
import re
import xml.etree.ElementTree as Xml
from pathlib import Path

# https://maven.apache.org/plugins/maven-checkstyle-plugin/history.html
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
DEPENDENCY_REGEX = re.compile(r"\$\{[^}]+}")


class MavenException(Exception):
    """
    Exception thrown whenever something wrong with pom.xml.
    """


def _find_pom_xml(project_dir: Path) -> Path:
    for subdir, _, files in os.walk(project_dir):
        if POM_XML in files:
            return Path(os.path.join(subdir, POM_XML))
    raise MavenException(f"No {POM_XML} detected in project.")


def _get_checkstyle_version_from_pom(pom: Path) -> str:
    if not str(pom).endswith(POM_XML):
        raise ValueError(f"No {POM_XML} was given")
    parsed_pom = Xml.parse(pom)
    root = parsed_pom.getroot()
    namespaces = _get_namespaces(root)

    # Find all plugin candidates
    plugin_candidates = []
    for plugin in root.findall(".//xmlns:plugin", namespaces=namespaces):
        artifact_id = plugin.find("xmlns:artifactId", namespaces=namespaces)
        if artifact_id is None:
            continue

        name = artifact_id.text
        if name == CHECKSTYLE_PLUGIN_ARTIFACT_ID:
            plugin_candidates.append(plugin)

    if len(plugin_candidates) == 0:
        raise MavenException("Project does not support checkstyle.")

    # Try all found plugins before raising an exception
    for plugin in plugin_candidates:
        version = _parse_checkstyle_version_from_xml_elem(plugin, root, namespaces)
        if version:
            return version

    raise MavenException("Not supported checkstyle version.")


def _get_namespaces(root: Xml.Element) -> dict[str, str]:
    namespaces = re.match(r"{(.*)}", root.tag)
    return {"xmlns": namespaces.group(1) if namespaces else STANDARD_NAMESPACE}


def _parse_checkstyle_version_from_xml_elem(
    elem: Xml.Element, root, namespaces: dict[str, str]
) -> None | str:
    # Find checkstyle version in artifact "checkstyle"
    checkstyle_dependencies = elem.findall(".//xmlns:dependency", namespaces=namespaces)
    for dependency in checkstyle_dependencies:
        name = dependency.find("xmlns:artifactId", namespaces=namespaces).text
        if name == CHECKSTYLE_ARTIFACT_ID:
            version = dependency.find("xmlns:version", namespaces=namespaces).text
            if version:
                if DEPENDENCY_REGEX.match(version):
                    version = _parse_checkstyle_version_from_variable(
                        version, root, namespaces
                    )
                return version

    # Get checkstyle version from maven-checkstyle-plugin version
    plugin_version = elem.find("xmlns:version", namespaces=namespaces)
    if plugin_version is not None:
        plugin_version = plugin_version.text
        if plugin_version:
            if DEPENDENCY_REGEX.match(plugin_version):
                plugin_version = _parse_checkstyle_version_from_variable(
                    plugin_version, root, namespaces
                )
            if plugin_version in MAVEN_PLUGIN_CHECKSTYLE_VERSION:
                return MAVEN_PLUGIN_CHECKSTYLE_VERSION[plugin_version]

    # No checkstyle version found
    return None


def get_checkstyle_version_of_project(project_dir: Path) -> str:
    """
    Returns the checkstyle version specified in pom.
    :param project_dir: Directory of the project.
    :return: Return the checkstyle version.
    """
    pom = _find_pom_xml(project_dir)
    return _get_checkstyle_version_from_pom(pom)


def _parse_checkstyle_version_from_variable(variable, root, namespaces):
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
