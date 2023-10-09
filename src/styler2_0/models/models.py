import json
from enum import Enum
from pathlib import Path
from typing import Any

from bidict import bidict
from streamerate import stream
from torch import Tensor, long, nn, tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from styler2_0.models.lstm import LSTM
from styler2_0.models.model_base import ModelBase
from styler2_0.utils.utils import read_content_of_file

TRAIN_DATA = Path("train")
TRAIN_SRC = TRAIN_DATA / Path("input.txt")
TRAIN_TRG = TRAIN_DATA / Path("ground_truth.txt")
VAL_DATA = Path("val")
VAL_SRC = VAL_DATA / Path("input.txt")
VAL_TRG = VAL_DATA / Path("ground_truth.txt")
SRC_VOCAB_FILE = Path("src_vocab.txt")
TRG_VOCAB_FILE = Path("trg_vocab.txt")
LSTM_INPUT_DIM = 650
LSTM_OUTPUT_DIM = 105


class Models(Enum):
    """
    The supported models.
    """

    LSTM = LSTM
    ANN = None
    NGRAM = None
    TRANSFORMER = None


# noinspection PyTypeChecker
def _load_train_and_val_data(
    project_dir: Path, src_vocab: bidict, trg_vocab: bidict, model: ModelBase
) -> tuple[DataLoader, DataLoader]:
    """
    Load the train and validation data.
    :param project_dir: The project directory.
    :param src_vocab: The source vocab.
    :param trg_vocab:The target vocab.
    :param model: The model.
    :return: The train and validation data.
    """
    train_inp = _load_file_to_tensor(
        project_dir / TRAIN_SRC, model.input_length, src_vocab
    )
    train_trg = _load_file_to_tensor(
        project_dir / TRAIN_TRG, model.output_length, trg_vocab
    )
    val_inp = _load_file_to_tensor(project_dir / VAL_SRC, model.input_length, src_vocab)
    val_trg = _load_file_to_tensor(
        project_dir / VAL_TRG, model.output_length, trg_vocab
    )
    return (
        DataLoader(list(zip(train_inp, train_trg, strict=True)), batch_size=3),
        DataLoader(list(zip(val_inp, val_trg, strict=True)), batch_size=2),
    )


def _load_file_to_tensor(file_path: Path, dim: int, vocab: bidict) -> Tensor:
    """
    Load the given file to a tensor.
    :param file_path: The given file path.
    :param dim: The dimension of the tensor.
    :param vocab: The vocab.
    :return: The tensor.
    """
    content = read_content_of_file(file_path)
    lines = content.split("\n")
    loaded_ints = list(
        stream(lines).map(lambda line: _line_to_tensor(line, dim, vocab)).to_list()
    )
    return tensor(loaded_ints, dtype=long)


def _line_to_tensor(line: str, dim: int, vocab: bidict) -> list:
    """
    Load the given line to a tensor.
    :param line: The given line.
    :param dim: The dimension of the tensor.
    :param vocab: The vocab.
    :return: The tensor.
    """
    raw_tensor = list(stream(line.split(" ")).map(int).to_list())
    raw_tensor.extend([vocab.inverse["<PAD>"]] * (dim - len(raw_tensor)))
    return raw_tensor


def _load_vocabs(project_dir: Path) -> tuple[bidict, bidict]:
    """
    Load the vocabs from the given project dir.
    :param project_dir: The given project dir.
    :return: The vocabs.
    """
    src_vocab = json.loads(
        read_content_of_file(project_dir / SRC_VOCAB_FILE),
        object_hook=lambda data: _load_vocab_to_dict_int_str(data),
    )
    trg_vocab = json.loads(
        read_content_of_file(project_dir / TRG_VOCAB_FILE),
        object_hook=lambda data: _load_vocab_to_dict_int_str(data),
    )
    return src_vocab, trg_vocab


def _load_vocab_to_dict_int_str(raw_vocab: Any) -> bidict[int, str]:
    """
    Load the given vocab to a bidict.
    :param raw_vocab: The given vocab.
    :return: The bidict.
    """
    return bidict({int(key): value for key, value in raw_vocab.items()})


def train(model: Models, project_dir: Path, epochs: int) -> None:
    """
    Train the given model.
    :param model: The given model.
    :param project_dir: Project dir with the data.
    :param epochs: Count epochs.
    :return:
    """

    # TODO:
    #    - Batch sizes as cl args
    #    - Criterion and optimizer from config
    model = model.value.build_from_config()
    src_vocab, trg_vocab = _load_vocabs(project_dir)
    train_data, val_data = _load_train_and_val_data(
        project_dir, src_vocab, trg_vocab, model
    )
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.inv["<PAD>"])
    optimizer = Adam(model.parameters())
    model.fit(epochs, train_data, val_data, criterion, optimizer)
    model.predict(
        read_content_of_file(
            Path(
                "/Users/maxij/PycharmProjects/styler2.0/data/tmp/model_data/three_gram/train/1/VIOLATED_NoteItemTriageStatusComparator.java"
            )
        ),
        Path("/Users/maxij/PycharmProjects/styler2.0/data/tmp/checkstyle.xml"),
        "8.0",
    )