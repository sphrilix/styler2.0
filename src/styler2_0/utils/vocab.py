import json
from pathlib import Path

from bidict import bidict

from src.styler2_0.utils.utils import read_content_of_file


class Vocabulary:
    """
    The vocabulary class.
    """

    def __init__(
        self,
        vocab: bidict[int, str],
        sos: str = "<SOS>",
        eos: str = "<EOS>",
        pad: str = "<PAD>",
        unk: str = "<UNK>",
    ) -> None:
        self._vocab = vocab
        self._sos = sos
        self._eos = eos
        self._pad = pad
        self._unk = unk

    def __len__(self) -> int:
        return len(self._vocab)

    def __getitem__(self, item: int | str) -> int | str:
        try:
            if isinstance(item, str):
                return self._vocab.inverse[item]
            return self._vocab[item]
        except KeyError:
            return self._vocab.inverse[self._unk]

    def stoi(self, item: str) -> int:
        """
        Convert the given item to an integer.
        :param item: The given item.
        :return: Returns the integer.
        """
        return self[item]

    def itos(self, item: int) -> str:
        """
        Convert the given item to a string.
        :param item: The given item.
        :return: Returns the string.
        """
        return self[item]

    @property
    def sos(self) -> str:
        return self._sos

    @property
    def eos(self) -> str:
        return self._eos

    @property
    def pad(self) -> str:
        return self._pad

    @property
    def unk(self) -> str:
        return self._unk

    @staticmethod
    def load(path: Path) -> "Vocabulary":
        """
        Load the vocabulary from the given path.
        :param path: The path where the vocab is stored.
        :return: Returns the loaded vocabulary.
        """
        return Vocabulary(
            bidict(
                json.loads(
                    read_content_of_file(path),
                    object_hook=lambda data: bidict(
                        {int(key): value for key, value in data.items()}
                    ),
                )
            )
        )

    def to_json(self) -> str:
        """
        Convert the vocabulary to a JSON string.
        :return: Returns the JSON string.
        """
        return json.dumps({str(k): v for k, v in self._vocab.items()})
