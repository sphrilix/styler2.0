import os
from math import isclose
from pathlib import Path
from random import shuffle
from shutil import copytree

from bidict import bidict
from streamerate import stream

from src.styler2_0.preprocessing.violation_generation import Metadata
from src.styler2_0.utils.utils import read_content_of_file, save_content_to_file

TRAIN_PATH = Path("train/")
VAL_PATH = Path("val/")
TEST_PATH = Path("test/")
DATA_JSON = Path("data.json")
VOCAB_FILE = Path("vocab.txt")
INPUT_TXT = Path("input.txt")
GROUND_TRUTH_TXT = Path("ground_truth.txt")
MODEL_DATA_PATH = Path("../../model-data")
VOCAB_SPECIAL_TOKEN = ["<GO>", "<EOS>", "<UNK>", "<PAD>"]


def _build_splits(violation_dir: Path, splits: (float, float, float)) -> None:
    if not isclose(1.0, sum(splits)):
        raise ValueError("Splits must sum to 1.0.")

    protocol = _get_protocol_from_path(violation_dir)
    dirs = os.listdir(violation_dir)
    shuffle(dirs)

    complete_train = violation_dir / MODEL_DATA_PATH / protocol / TRAIN_PATH
    complete_val = violation_dir / MODEL_DATA_PATH / protocol / VAL_PATH
    complete_test = violation_dir / MODEL_DATA_PATH / protocol / TEST_PATH
    os.makedirs(complete_train, exist_ok=True)
    os.makedirs(complete_val, exist_ok=True)
    os.makedirs(complete_test, exist_ok=True)

    # Train = [0, train_percentile)
    # Val = [train_percentile, train_percentile + val_percentile)
    # Test = [train_percentile + val_percentile, 1.0]
    train_part = splits[0]
    val_part = train_part + splits[1]
    training = dirs[: int(len(dirs) * train_part)]
    validation = dirs[int(len(dirs) * train_part) : int(len(dirs) * val_part)]
    testing = dirs[int(len(dirs) * val_part) :]
    for violation in training:
        copytree(violation_dir / Path(violation), complete_train / Path(violation))
    for violation in validation:
        copytree(violation_dir / Path(violation), complete_val / Path(violation))
    for violation in testing:
        copytree(violation_dir / Path(violation), complete_test / Path(violation))


def _build_vocab(violation_dir: Path) -> bidict[int, str]:
    build_vocab_path = (
        violation_dir
        / MODEL_DATA_PATH
        / _get_protocol_from_path(violation_dir)
        / TRAIN_PATH
    )
    metadata = VOCAB_SPECIAL_TOKEN
    for violation in os.listdir(build_vocab_path):
        metadata_json_content = read_content_of_file(
            build_vocab_path / Path(violation) / DATA_JSON
        )
        metadata.append(Metadata.from_json(metadata_json_content))
    vocab_dict = bidict(
        stream(metadata)
        .flatMap(lambda md: _get_vocabs_from_metadata(md))
        .to_set()
        .enumerate()
        .to_dict()
    )
    save_content_to_file(
        violation_dir
        / MODEL_DATA_PATH
        / _get_protocol_from_path(violation_dir)
        / VOCAB_FILE,
        str(vocab_dict),
    )
    return vocab_dict


def _build_inputs_from_vocab(input_dir: Path, vocab: bidict) -> None:
    model_input = []
    ground_truth = []
    for violation in os.listdir(input_dir):
        metadata = Metadata.from_json(
            read_content_of_file(input_dir / violation / DATA_JSON)
        )
        violated_tokens = metadata.violated_str.split(" ")
        non_violated_tokens = metadata.non_violated_str.split(" ")
        violated_ids = " ".join(
            str(_map_token_to_id(token, vocab)) for token in violated_tokens
        )
        non_violated_ids = " ".join(
            str(_map_token_to_id(token, vocab)) for token in non_violated_tokens
        )
        model_input.append(violated_ids)
        ground_truth.append(non_violated_ids)
    save_content_to_file(input_dir / INPUT_TXT, "\n".join(model_input))
    save_content_to_file(input_dir / GROUND_TRUTH_TXT, "\n".join(ground_truth))


def _map_token_to_id(token: str, vocab: bidict) -> int:
    if token in vocab.inverse:
        return vocab.inverse[token]
    return vocab.inverse["<UNK>"]


def _get_vocabs_from_metadata(metadata: Metadata | str) -> list[str]:
    if isinstance(metadata, str):
        return [metadata]
    violated_vocab = metadata.violated_str.split(" ")
    non_violated_vocab = metadata.non_violated_str.split(" ")
    return violated_vocab + non_violated_vocab


def _get_protocol_from_path(violation_dir: Path) -> Path:
    return Path(violation_dir.name)


def preprocessing(violation_dir: Path, splits: (float, float, float)) -> None:
    """
    Build the input for the lstm model.
    :param violation_dir: The directory containing the violations
    :param splits: Triple(train_percentile, val_percentile, test_percentile)
    :return:
    """
    _build_splits(violation_dir, splits)
    vocab = _build_vocab(violation_dir)
    _build_inputs_from_vocab(
        violation_dir
        / MODEL_DATA_PATH
        / _get_protocol_from_path(violation_dir)
        / TRAIN_PATH,
        vocab,
    )
    _build_inputs_from_vocab(
        violation_dir
        / MODEL_DATA_PATH
        / _get_protocol_from_path(violation_dir)
        / VAL_PATH,
        vocab,
    )
    _build_inputs_from_vocab(
        violation_dir
        / MODEL_DATA_PATH
        / _get_protocol_from_path(violation_dir)
        / TEST_PATH,
        vocab,
    )
