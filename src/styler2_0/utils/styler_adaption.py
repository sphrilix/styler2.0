import csv
from pathlib import Path

STYLER_TOKEN_MAPPING = {
    "Annotation": "AT",
    "Boolean": "BOOL_LITERAL",
    "OctalInteger": "OCTAL_LITERAL",
    "DecimalInteger": "INTEGER_LITERAL",
    "BinaryInteger": "BINARY_LITERAL",
    "HexInteger": "HEX_LITERAL",
    "DecimalFloatingPoint": "DECIMAL_LITERAL",
    "Comment": "COMMENT",
    "Null": "NULL_LITERAL",
    "String": "STRING_LITERAL",
    "Char": "CHAR_LITERAL",
    "Identifier": "IDENTIFIER",
}


def adapt_styler_three_gram_csv(in_file: Path, out_file: Path) -> None:
    """
    Adapt the styler 3-gram csv to match the tokens of this preprocessing.
    :param in_file: The styler csv.
    :param out_file: Where to store the new csv.
    :return:
    """
    if not str(in_file).endswith(".csv") or not str(out_file).endswith(".csv"):
        raise ValueError("Not a csv file supplied.")
    new_rows = []
    with open(in_file) as file_stream:
        csv_rows = csv.reader(file_stream, quoting=csv.QUOTE_MINIMAL)
        for row in csv_rows:
            new_row = [
                _adapt_token(row[0]),
                _adapt_ws(row[1]),
                _adapt_token(row[2]),
                row[3],
            ]
            new_rows.append(new_row)
    with open(out_file, "w") as out_stream:
        writer = csv.writer(out_stream)
        writer.writerows(new_rows)


def _adapt_three_gram(token1: str, ws: str, token2: str) -> [str, str, str]:
    return _adapt_token(token1), _adapt_ws(ws), _adapt_token(token2)


def _adapt_ws(ws: str) -> str:
    return ws.replace("ID_", "").replace("DD_", "")


def _adapt_token(token: str) -> str:
    if token in STYLER_TOKEN_MAPPING:
        return STYLER_TOKEN_MAPPING[token]
    return token
