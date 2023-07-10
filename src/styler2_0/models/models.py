import json
from enum import Enum
from pathlib import Path
from typing import Any

from bidict import bidict
from streamerate import stream
from torch import Tensor, long, nn, tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from styler2_0.models.lstm import LSTM, LSTMDecoder, LSTMEncoder
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
    LSTM = "LSTM"
    ANN = "ANN"
    NGRAM = "NGRAM"
    TRANSFORMER = "TRANSFORMER"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        raise ValueError(f"{value} is not a supported model.")


def _build_model(model: Models) -> ModelBase:
    match model:
        case Models.LSTM:
            return _build_lstm()
        case _:
            raise ValueError()


# noinspection PyTypeChecker
def _load_train_and_val_data(
    project_dir: Path, src_vocab: bidict, trg_vocab: bidict
) -> tuple[DataLoader, DataLoader]:
    train_inp = _load_file_to_tensor(project_dir / TRAIN_SRC, LSTM_INPUT_DIM, src_vocab)
    train_trg = _load_file_to_tensor(
        project_dir / TRAIN_TRG, LSTM_OUTPUT_DIM, trg_vocab
    )
    val_inp = _load_file_to_tensor(project_dir / VAL_SRC, LSTM_INPUT_DIM, src_vocab)
    val_trg = _load_file_to_tensor(project_dir / VAL_TRG, LSTM_OUTPUT_DIM, trg_vocab)
    return (
        DataLoader(list(zip(train_inp, train_trg, strict=True)), batch_size=3),
        DataLoader(list(zip(val_inp, val_trg, strict=True)), batch_size=2),
    )


def _load_file_to_tensor(file_path: Path, dim: int, vocab: bidict) -> Tensor:
    content = read_content_of_file(file_path)
    lines = content.split("\n")
    loaded_ints = list(
        stream(lines).map(lambda line: _line_to_tensor(line, dim, vocab)).to_list()
    )
    return tensor(loaded_ints, dtype=long)


def _line_to_tensor(line: str, dim: int, vocab: bidict) -> list:
    raw_tensor = list(stream(line.split(" ")).map(int).to_list())
    raw_tensor.extend([vocab.inverse["<PAD>"]] * (dim - len(raw_tensor)))
    return raw_tensor


def _load_vocabs(project_dir: Path) -> tuple[bidict, bidict]:
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
    return bidict({int(key): value for key, value in raw_vocab.items()})


def _build_lstm() -> LSTM:
    enc_emb_dim = 256
    dec_emb_dim = 256
    hid_dim = 512
    n_layers = 1
    enc_dropout = 0.5
    dec_dropout = 0.5

    device = "cpu"

    enc = LSTMEncoder(LSTM_INPUT_DIM, enc_emb_dim, hid_dim, n_layers, enc_dropout)
    dec = LSTMDecoder(LSTM_OUTPUT_DIM, dec_emb_dim, hid_dim, n_layers, dec_dropout)

    return LSTM(enc, dec, device).to(device)


def train(model: Models, project_dir: Path, epochs: int) -> None:
    """
    Train the given model.
    :param model: The given model.
    :param project_dir: Project dir with the data.
    :param epochs: Count epochs.
    :return:
    """
    model = _build_model(model)
    src_vocab, trg_vocab = _load_vocabs(project_dir)
    train_data, val_data = _load_train_and_val_data(project_dir, src_vocab, trg_vocab)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.inv["<PAD>"])
    optimizer = Adam(model.parameters())
    model.fit(epochs, train_data, val_data, criterion, optimizer)
