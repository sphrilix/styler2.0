import sys
from typing import List

from utils.tokenize import lex_java


def _main(args: List[str]) -> int:
    print(args)
    return 0


if __name__ == "__main__":
    tokens = lex_java(
        "public class Main {"
        "   public static void main(String[] args) {"
        '         System.out.println("test");\n'
        "   \t}"
        "}"
    )
    for token in tokens:
        print(token)
