import logging
import os

import javalang

from src.styler2_0.main import setup_logging

INPUT_DIR = r"D:\PyCharm_Projects_D\styler2.0\Tmp"
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


def _iterate_methods(file: str) -> dict:
    """
    Iterates over the methods in a file and returns a dictionary containing the method
    name and the method code.
    :param file: The file.
    :return: A dictionary containing the method name and the method code.
    """
    with open(file) as r:
        codelines = r.readlines()
        code_text = "".join(codelines)

    lex = None
    parse_tree = javalang.parse.parse(code_text)
    methods = {}
    for _, method_node in parse_tree.filter(javalang.tree.MethodDeclaration):
        startpos, endpos, startline, endline = get_method_start_end(
            parse_tree, method_node
        )
        method_text, startline, endline, lex = get_method_text(
            codelines, startpos, endpos, startline, endline, lex
        )
        methods[method_node.name] = method_text

    return methods


def get_method_start_end(parse_tree, method_node):
    startpos = None
    endpos = None
    startline = None
    endline = None
    for path, node in parse_tree:
        if startpos is not None and method_node not in path:
            endpos = node.position
            endline = node.position.line if node.position is not None else None
            break
        if startpos is None and node == method_node:
            startpos = node.position
            startline = node.position.line if node.position is not None else None
    return startpos, endpos, startline, endline


def get_method_text(
    codelines, startpos, endpos, startline, endline, last_endline_index
):
    if startpos is None:
        return "", None, None, None

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
        _extract_methods_from_dir(os.path.join(INPUT_DIR, directory), OUTPUT_DIR)


if __name__ == "__main__":
    main()
