import logging
import os
from typing import Any

import javalang
from javalang.tree import MethodDeclaration

from src.styler2_0.main import setup_logging

INPUT_DIR = r"D:\PyCharm_Projects_D\styler2.0\extracted"
OUTPUT_DIR = r"D:\PyCharm_Projects_D\styler2.0\methods"


def _extract_methods_from_dir(input_dir: str, output_dir: str) -> None:
    """
    Extracts java methods from their classes and stores each in a separate file.
    :param input_dir: The input directory.
    :param output_dir: The output directory.
    :return: None.
    """
    # Iterate over each file in the input directory
    for file in os.listdir(input_dir):
        file = os.path.join(input_dir, file)
        # Check if the file is a file and if it has the correct file type
        if os.path.isfile(file) and file.endswith(".java"):
            _extract_methods_from_file(file, output_dir)
        else:
            logging.warning("File %s is not a java file.", file)


def _extract_methods_from_file(input_file: str, output_dir: str) -> None:
    """
    Extracts java methods from a file and stores each in a separate file.
    :param input_file: The input file.
    :param output_dir: The output directory.
    :return: None.
    """
    # Extract the methods from the source code
    methods = _iterate_methods(input_file)
    logging.info("Found %d methods in file %s.", len(methods), input_file)

    # Create a subfolder for each input file in the output directory
    output_subdir = os.path.join(output_dir, os.path.basename(input_file))
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
    # Read the file
    with open(file) as r:
        codelines = r.readlines()
        code_text = "".join(codelines)

    methods = {}

    # Try to parse the file
    lex = None
    try:
        parse_tree = javalang.parse.parse(code_text)
    except javalang.parser.JavaSyntaxError as e:
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
    Get the text of a method.
    :param codelines: The code lines.
    :param startpos: The start position of the method.
    :param endpos: The end position of the method.
    :param startline: The start line of the method.
    :param endline: The end line of the method.
    :param last_endline_index: The index of the last end line.
    :return: The text of the method.
    """
    if startpos is None:
        return "", None, None, None

    # Get the start and end line index
    startline_index = startline - 1
    endline_index = endline - 1 if endpos is not None else None

    # 1. check for and fetch annotations
    if last_endline_index is not None:
        for line in codelines[(last_endline_index + 1) : (startline_index)]:
            if "@" in line:
                startline_index = startline_index - 1
    meth_text = "<ST>".join(codelines[startline_index:endline_index])
    meth_text = meth_text[: meth_text.rfind("}") + 1]

    # 2. remove trailing rbrace for last methods & any external content/comments
    # if endpos is None and
    if abs(meth_text.count("}") - meth_text.count("{")) != 0:
        # imbalanced braces
        brace_diff = abs(meth_text.count("}") - meth_text.count("{"))

        for _ in range(brace_diff):
            meth_text = meth_text[: meth_text.rfind("}")]
            meth_text = meth_text[: meth_text.rfind("}") + 1]

    # 3. remove any trailing comments
    meth_lines = meth_text.split("<ST>")
    meth_text = "".join(meth_lines)
    last_endline_index = startline_index + (len(meth_lines) - 1)

    return (
        meth_text,
        (startline_index + 1),
        (last_endline_index + 1),
        last_endline_index,
    )


def main():
    """
    Extracts java methods from their classes and stores each in a separate file.
    """
    # Setup logging
    log_file = os.path.join(OUTPUT_DIR, "method_extractor.log")
    setup_logging(log_file)

    # Iterate over each directory in the input directory
    for directory in os.listdir(INPUT_DIR):
        # Create a subfolder for each directory in the output directory
        output_subdir = os.path.join(OUTPUT_DIR, directory)
        _extract_methods_from_dir(os.path.join(INPUT_DIR, directory), output_subdir)


if __name__ == "__main__":
    main()
