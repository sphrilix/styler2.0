import logging
import os
import sys
from enum import Enum
from typing import Any

import javalang
from javalang.tree import MethodDeclaration

from src.styler2_0.main import setup_logging


class OverwriteMode(Enum):
    """
    Enum for the overwrite mode.
    """

    OVERWRITE = 0
    SKIP = 1


# TODO: Fix Abstract methods of interfaces e.g. hadoop\AbfsCounter.java

INPUT_DIR = r"D:\PyCharm_Projects_D\styler2.0\extracted"
OUTPUT_DIR = r"D:\PyCharm_Projects_D\styler2.0\methods_2"
OVERWRITE_MODE = OverwriteMode.SKIP
INCLUDE_METHOD_COMMENTS = True
COMMENTS_REQUIRED = True
REMOVE_INDENTATION = True


def _extract_methods_from_dir(input_dir: str, output_dir: str) -> None:
    """
    Extracts java methods from their classes and stores each in a separate file.
    :param input_dir: The input directory.
    :param output_dir: The output directory.
    :return: None.
    """
    # Check if the input directory exists and is a directory
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        logging.error(
            "Input directory %s does not exist or is not a directory.", input_dir
        )
        return

    # Iterate over each file in the input directory
    for file in os.listdir(input_dir):
        file = os.path.join(input_dir, file)
        # Check if the file is a file and if it has the correct file type
        if os.path.isfile(file) and file.endswith(".java"):
            _extract_methods_from_file(file, output_dir)
        else:
            logging.warning("File %s is not a java file.", file)


def _extract_methods_from_file(
    input_file: str,
    output_dir: str,
) -> None:
    """
    Extracts java methods from a file and stores each in a separate file.
    :param input_file: The input file.
    :param output_dir: The output directory.
    :return: None.
    """
    # Check if the input file exists and is a file
    if not os.path.exists(input_file) or not os.path.isfile(input_file):
        logging.error("Input file %s does not exist or is not a file.", input_file)
        return

    # Specify the output subdirectory path and name
    output_subdir = os.path.join(output_dir, os.path.basename(input_file))

    # Check if we skip the file
    if os.path.exists(output_subdir) and OVERWRITE_MODE == OverwriteMode.SKIP:
        logging.info(
            "Skipping file %s, because the output directory %s already exists.",
            input_file,
            output_dir,
        )
        return

    # Extract the methods from the source code
    methods = _iterate_methods(input_file)
    logging.info("Found %d methods in file %s.", len(methods), input_file)

    # Create a subfolder for each input file if methods were found
    if methods:
        os.makedirs(output_subdir, exist_ok=True)

    # Write each method to a separate file
    for method_name, method_code in methods.items():
        output_file = os.path.join(output_subdir, method_name + ".java")
        with open(output_file, "w") as w:
            w.write(method_code)


def _iterate_methods(file: str) -> dict[str, str]:
    """
    Iterates over the methods in a file and returns a dictionary containing the method
    name and the method code.
    :param file: The file.
    :return: A dictionary containing the method name and the method code.
    """
    # Check if the file exists and is a java file
    if not os.path.exists(file) or not file.endswith(".java"):
        logging.error("File %s does not exist or is not a java file.", file)
        return {}

    # Read the file
    try:
        with open(file) as r:
            codelines = r.readlines()
            code_text = "".join(codelines)
    except UnicodeDecodeError as e:
        logging.warning("Could not read file %s.", file)
        logging.warning(e)
        return {}

    methods = {}

    # Try to parse the file
    lex = None
    try:
        parse_tree = javalang.parse.parse(code_text)
    except Exception as e:
        logging.warning("Could not parse file %s.", file)
        logging.warning(e)
        return {}

    for _, method_node in parse_tree.filter(MethodDeclaration):
        startpos, endpos, startline, endline = _get_method_start_end(
            parse_tree, method_node
        )
        method_text, startline, endline, lex = _get_method_text(
            codelines, startpos, endpos, startline, endline, lex
        )

        # Get the first line of the method text
        first_line = method_text.split("\n")[0].strip()

        # Check if COMMENTS_REQUIRED is True and the method has a comment
        if COMMENTS_REQUIRED and not first_line.startswith("/"):
            logging.info(
                "Skipping method %s, because it has no comment.", method_node.name
            )
            continue

        methods[method_node.name] = method_text

    return methods


def _get_method_start_end(
    parse_tree: list, method_node: MethodDeclaration
) -> tuple[str, str, int, int]:
    """
    Get the start and end position of a method in the source code.
    :param parse_tree: The full parse tree.
    :param method_node: The method node.
    :return: The start and end position of the method.
    """
    startpos = None
    endpos = None
    startline = None
    endline = None

    # Iterate over the parse tree and find the method node
    for path, node in parse_tree:
        if startpos is not None and method_node not in path:
            endpos = node.position
            endline = node.position.line if node.position is not None else None
            break
        if startpos is None and node == method_node:
            startpos = node.position
            startline = node.position.line if node.position is not None else None
    return startpos, endpos, startline, endline


def _get_method_text(
    codelines: list[str],
    startpos: str,
    endpos: str,
    startline: int,
    endline: int,
    last_endline_index: int,
) -> tuple[str, int | None, int | None, Any]:
    """
    Get the text of a method, including any comments before the method.
    :param codelines: The code lines.
    :param startpos: The start position of the method.
    :param endpos: The end position of the method.
    :param startline: The start line of the method.
    :param endline: The end line of the method.
    :param last_endline_index: The comment_index of the last end line.
    :return: The text of the method.
    """
    if startpos is None:
        return "", None, None, None

    # Get the start and end line comment_index
    startline_index = startline - 1
    endline_index = endline - 1 if endpos is not None else None

    # Fetch the method code
    meth_text = "<ST>".join(codelines[startline_index:endline_index])
    meth_text = meth_text[: meth_text.rfind("}") + 1]

    # Remove trailing rbrace for last methods & any external content/comments
    # if endpos is None and
    if abs(meth_text.count("}") - meth_text.count("{")) != 0:
        # imbalanced braces
        brace_diff = abs(meth_text.count("}") - meth_text.count("{"))

        for _ in range(brace_diff):
            meth_text = meth_text[: meth_text.rfind("}")]
            meth_text = meth_text[: meth_text.rfind("}") + 1]

    # Remove any trailing comments within the method
    meth_lines = meth_text.split("<ST>")
    meth_text = "".join(meth_lines)
    last_endline_index = startline_index + (len(meth_lines) - 1)

    # Include comments before the method
    if last_endline_index is not None:
        comment_lines = []
        comment_index = startline_index - 1
        while comment_index >= 0 and (
            codelines[comment_index].strip().startswith("/")
            or codelines[comment_index].strip().startswith("*")
        ):
            comment_lines.insert(0, codelines[comment_index])
            comment_index -= 1

        comment_block = "".join(comment_lines)

        # Include comments block at the beginning of the method text
        if comment_block and INCLUDE_METHOD_COMMENTS:
            meth_text = comment_block + meth_text
            startline_index = comment_index + 1

    # Remove indentation from the method text
    if REMOVE_INDENTATION:
        meth_text = _remove_indentation(meth_text)

    return (
        meth_text,
        (startline_index + 1),
        (last_endline_index + 1),
        last_endline_index,
    )


def _remove_indentation(meth_text: str) -> str:
    """
    Remove indentation from the method text there is the same amount of indentation
    on each line.
    :param meth_text: The method text.
    :return: The method text without indentation.
    """
    meth_lines = meth_text.split("\n")
    indentation_changed = False
    minimal_indentation = len(meth_lines[0]) - len(meth_lines[0].lstrip())

    # Calculate the indentation of all lines
    for line in meth_lines:
        # Skip empty lines
        if not line.strip():
            continue

        indentation = len(line) - len(line.lstrip())

        if indentation < minimal_indentation:
            indentation_changed = True
            break

        minimal_indentation = min(indentation, minimal_indentation)

    # If the indentation is the same on each line, remove it
    if not indentation_changed:
        indentation = minimal_indentation
        for i, line in enumerate(meth_lines):
            meth_lines[i] = line[indentation:]

    return "\n".join(meth_lines)


def main():
    """
    Extracts java methods from their classes and stores each in a separate file.
    """
    # Setup logging
    log_file = os.path.join(OUTPUT_DIR, "method_extractor.log")
    setup_logging(log_file)

    # Don't allow COMMENTS_REQUIRED to be True if INCLUDE_METHOD_COMMENTS is False
    if COMMENTS_REQUIRED and not INCLUDE_METHOD_COMMENTS:
        logging.error(
            "INCLUDE_METHOD_COMMENTS must be True if COMMENTS_REQUIRED is True."
        )
        sys.exit(1)

    # Iterate over each directory in the input directory
    for directory in os.listdir(INPUT_DIR):
        if not os.path.isdir(os.path.join(INPUT_DIR, directory)):
            continue

        # Create a subfolder for each directory in the output directory
        output_subdir = os.path.join(OUTPUT_DIR, directory)
        _extract_methods_from_dir(os.path.join(INPUT_DIR, directory), output_subdir)


if __name__ == "__main__":
    main()
