from pathlib import Path

from src.styler2_0.utils.tokenize import ProcessedToken, Identifier, ProcessedSourceFile, Whitespace


def test_pad_source_file() -> None:
    tokens: list[ProcessedToken] = [Identifier("Test", 0, 0), Identifier("Test2", 0, 4)]
    processed_file = ProcessedSourceFile(Path("."), tokens)
    assert len(processed_file.tokens) == 4
    assert isinstance(processed_file.tokens[1], Whitespace)
    assert isinstance(processed_file.tokens[3], Whitespace)
