import sys
from typing import List


def _main(args: List[str]) -> int:
    print(args)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
