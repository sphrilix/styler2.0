from pathlib import Path

from utils.checkstyle import run_checkstyle_on_dir
from utils.tokenize import tokenize_dir


def _main(args: list[str]) -> int:
    print(args)
    return 0


if __name__ == "__main__":
    print(tokenize_dir(Path("/Users/maxij/PycharmProjects/styler2.0/data")))
    print(
        run_checkstyle_on_dir(
            Path("/Users/maxij/PycharmProjects/styler2.0/data"), "8.0"
        )
    )
