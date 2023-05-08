from pathlib import Path

from utils.tokenize import tokenize_dir


def _main(args: list[str]) -> int:
    print(args)
    return 0


if __name__ == "__main__":
    print(tokenize_dir(Path("/Users/maxij/IdeaProjects/preprocessing-toolbox/src/main/java/de/uni_passau/fim/se2/deepcode/toolbox/ast/parser")))
