import re
from abc import ABC, abstractmethod


class ModelTokenizer(ABC):
    """
    Base class for model tokenizers.
    """

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize the given text.
        output: [max_len]
        :param text: The text to be prepared.
        :return: Returns the tokens.
        """
        return self._process_tokens_for_inp(self.get_tokens(text))

    @abstractmethod
    def get_tokens(self, text: str) -> list[str]:
        """
        Split the given text into tokens.
        :param text: The text to be split.
        :return: Returns the tokens.
        """
        pass

    @abstractmethod
    def _process_tokens_for_inp(self, tokens: list[str]) -> list[str]:
        """
        Process the tokens for the input.
        :param tokens: The tokens.
        :return: Returns the processed tokens.
        """
        pass


class SplitByTokenizer(ModelTokenizer):
    """
    Split token stream by certain char.
    """

    def __init__(
        self, max_length: int, split_by: str = " ", pad: str = "<PAD>"
    ) -> None:
        self._max_length = max_length
        self._split_by = split_by
        self._pad = pad

    def get_tokens(self, text: str) -> list[str]:
        """
        Split the given text into its tokens.
        :param text: Input text
        :return: Returns the tokens.
        """
        return text.split(self._split_by)

    def _process_tokens_for_inp(self, tokens: list[str]) -> list[str]:
        """
        Process the tokens for the input.
        :param tokens: The tokens from get_tokens.
        :return: Returns a list of tokens prepared for model input.
        """
        tokens = tokens[: self._max_length]
        return tokens + [self._pad] * (self._max_length - len(tokens))


class SequenceTokenizer(SplitByTokenizer):
    """
    Tokenizer for Sequence Models like LSTM and Transformer.
    """

    def __init__(
        self,
        max_length: int,
        sos: str = "<SOS>",
        eos: str = "<EOS>",
        pad: str = "<PAD>",
    ):
        self._sos = sos
        self._eos = eos
        super().__init__(max_length, " ", pad)

    def _process_tokens_for_inp(self, tokens: list[str]) -> list[str]:
        """
        Process the tokens for the input.
        :param tokens: The tokens from get_tokens.
        :return: Returns a list of tokens prepared for model input.
        """
        tokens = [self._sos] + tokens[: self._max_length - 2] + [self._eos]
        tokens = tokens + [self._pad] * (self._max_length - len(tokens))
        return tokens


class NoneTokenizer(ModelTokenizer):
    """
    As the output vocab for the ANN is just the whole target string
    not split into sub-tokens and the preprocessing needs a tokenizer
    to be applicable. This Tokenizer just returns the unprocessed
    token str as expected by the preprocessing.
    """

    def get_tokens(self, text: str) -> list[str]:
        """
        Returns a one-element list of the input text.
        :param text: Input text
        :return: Returns [text]
        """
        return [text]

    def _process_tokens_for_inp(self, tokens: list[str]) -> list[str]:
        """
        No further processing is needed by the NoneTokenizer.
        Therefore, return its input.
        :param tokens: The input tokens.
        :return: Returns tokens.
        """
        return tokens


class SplitByCheckstyleTokenizer(ModelTokenizer):
    """
    Split token stream by checkstyle tags.
    """

    CHECKSTYLE_TOKEN_REG = re.compile(r"</?(\w+)>")

    def __init__(self, pad: str = "<PAD>") -> None:
        self._pad = pad

    def get_tokens(self, text: str) -> list[str]:
        """
        Split the given text into its tokens.
        :param text: Input text
        :return: Returns the tokens.
        """
        splits = re.split(self.CHECKSTYLE_TOKEN_REG, text)
        processed_middle_str = f"<{splits[1]}>{splits[2]}</{splits[1]}>"
        return [splits[0], processed_middle_str, splits[4]]

    def _process_tokens_for_inp(self, tokens: list[str]) -> list[str]:
        """
        Process the tokens for the input.
        :param tokens: The tokens from get_tokens.
        :return: Returns a list of tokens prepared for model input.
        """
        return tokens + [self._pad] * (3 - len(tokens))
