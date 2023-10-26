from abc import ABC, abstractmethod


class ModelTokenizer(ABC):
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize the given text.
        output: [seq_len]
        :param is_inp: specify whether the text is an input or an output.
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


class SequenceTokenizer(ModelTokenizer):
    def __init__(
        self,
        max_length: int,
        sos: str = "<SOS>",
        eos: str = "<EOS>",
        pad: str = "<PAD>",
    ):
        self._sos = sos
        self._eos = eos
        self._pad = pad
        self._max_length = max_length

    def get_tokens(self, text: str) -> list[str]:
        """
        Split the given text into tokens.
        :param text: The text to be split.
        :return: Returns the tokens.
        """
        return text.split(" ")

    def _process_tokens_for_inp(self, tokens: list[str]) -> list[str]:
        """
        Process the tokens for the input.
        :param tokens: The tokens from get_tokens.
        :return: Returns a list of tokens prepared for model input.
        """
        tokens = [self._sos] + tokens[: self._max_length - 2] + [self._eos]
        tokens = tokens + [self._pad] * (self._max_length - len(tokens))
        return tokens
