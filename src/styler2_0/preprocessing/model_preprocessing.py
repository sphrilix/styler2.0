import json
import os
from math import isclose
from pathlib import Path
from random import shuffle
from shutil import copytree

from bidict import bidict
from streamerate import stream

from src.styler2_0.models.models import Models
from src.styler2_0.preprocessing.model_tokenizer import (
    ModelTokenizer,
    SequenceTokenizer,
)
from src.styler2_0.preprocessing.violation_generation import Metadata
from src.styler2_0.utils.utils import read_content_of_file, save_content_to_file
from src.styler2_0.utils.vocab import Vocabulary

VIOLATION_DIR = Path("violations/")
TRAIN_PATH = Path("train/")
VAL_PATH = Path("val/")
TEST_PATH = Path("test/")
DATA_JSON = Path("data.json")
SRC_VOCAB_FILE = Path("src_vocab.txt")
TRG_VOCAB_FILE = Path("trg_vocab.txt")
INPUT_TXT = Path("input.txt")
GROUND_TRUTH_TXT = Path("ground_truth.txt")
MODEL_DATA_PATH = Path("../../model_data")
VOCAB_SPECIAL_TOKEN = ["<SOS>", "<EOS>", "<UNK>", "<PAD>"]


def _build_splits(violation_dir: Path, splits: (float, float, float)) -> None:
    if not isclose(1.0, sum(splits)):
        raise ValueError("Splits must sum to 1.0.")

    protocol = _get_protocol_from_path(violation_dir)
    dirs = os.listdir(violation_dir)
    shuffle(dirs)

    complete_train = violation_dir / MODEL_DATA_PATH / protocol / TRAIN_PATH
    complete_val = violation_dir / MODEL_DATA_PATH / protocol / VAL_PATH
    complete_test = violation_dir / MODEL_DATA_PATH / protocol / TEST_PATH

    # If the splits are already built, don't rebuild them.
    if complete_train.exists() and complete_val.exists() and complete_test.exists():
        return

    os.makedirs(complete_train)
    os.makedirs(complete_val)
    os.makedirs(complete_test)

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


def _build_vocab(
    violation_dir: Path,
    src_tokenizer: ModelTokenizer,
    trg_tokenizer: ModelTokenizer,
    model: Models,
) -> tuple[Vocabulary, Vocabulary]:
    build_vocabs_path = (
        violation_dir
        / MODEL_DATA_PATH
        / _get_protocol_from_path(violation_dir)
        / TRAIN_PATH
    )
    metadata = []
    for violation in os.listdir(build_vocabs_path):
        metadata_json_content = read_content_of_file(
            build_vocabs_path / Path(violation) / DATA_JSON
        )
        metadata.append(Metadata.from_json(metadata_json_content))
    src_vocab, trg_vocab = _get_vocabs_from_metadata(
        metadata, src_tokenizer, trg_tokenizer
    )

    save_path = (
        violation_dir
        / MODEL_DATA_PATH
        / _get_protocol_from_path(violation_dir)
        / model.name.lower()
    )

    os.makedirs(save_path, exist_ok=True)

    save_content_to_file(
        save_path / SRC_VOCAB_FILE,
        src_vocab.to_json(),
    )
    save_content_to_file(
        save_path / TRG_VOCAB_FILE,
        trg_vocab.to_json(),
    )
    return src_vocab, trg_vocab


def _build_inputs_from_vocab(
    input_dir: Path,
    output_dir: Path,
    src_vocab: Vocabulary,
    trg_vocab: Vocabulary,
    src_tokenizer: ModelTokenizer,
    trg_tokenizer: ModelTokenizer,
) -> None:
    model_input = []
    ground_truth = []
    for violation in os.listdir(input_dir):
        metadata = Metadata.from_json(
            read_content_of_file(input_dir / violation / DATA_JSON)
        )

        violated_tokens = src_tokenizer.tokenize(metadata.violated_str)
        non_violated_tokens = trg_tokenizer.tokenize(metadata.non_violated_str)

        violated_ids = " ".join(str(src_vocab[token]) for token in violated_tokens)
        non_violated_ids = " ".join(
            str(trg_vocab[token]) for token in non_violated_tokens
        )
        model_input.append(violated_ids)
        ground_truth.append(non_violated_ids)
    save_content_to_file(output_dir / INPUT_TXT, "\n".join(model_input))
    save_content_to_file(output_dir / GROUND_TRUTH_TXT, "\n".join(ground_truth))


def _get_vocabs_from_metadata(
    metadata: list[Metadata],
    src_tokenizer: ModelTokenizer,
    trg_tokenizer: ModelTokenizer,
) -> (Vocabulary, Vocabulary):
    src_vocab_tokens = []
    trg_vocab_tokens = []
    src_vocab_tokens.append(VOCAB_SPECIAL_TOKEN)
    trg_vocab_tokens.append(VOCAB_SPECIAL_TOKEN)

    # Introduce temporary sets to eliminate duplicates in vocabs and ensuring special
    # tokens have the same indexes in each vocab (src, trg).
    temp_src_vocab_tokens = set()
    temp_trg_vocab_tokens = set()
    for md in metadata:
        temp_src_vocab_tokens.update(src_tokenizer.get_tokens(md.violated_str))
        temp_trg_vocab_tokens.update(trg_tokenizer.get_tokens(md.non_violated_str))
    src_vocab_tokens.append(temp_src_vocab_tokens)
    trg_vocab_tokens.append(temp_trg_vocab_tokens)

    src_vocab = dict(stream(src_vocab_tokens).flatMap().enumerate().to_dict())
    trg_vocab = dict(stream(trg_vocab_tokens).flatMap().enumerate().to_dict())

    return Vocabulary(bidict(src_vocab)), Vocabulary(bidict(trg_vocab))


def _get_protocol_from_path(violation_dir: Path) -> Path:
    return Path(violation_dir.name)


def _build_model_tokenizers(model: Models) -> tuple[ModelTokenizer, ModelTokenizer]:
    model_data = model.value.get_model_params()
    output_length = model_data["output_length"]
    input_length = model_data["input_length"]
    match model:
        case Models.LSTM | Models.TRANSFORMER:
            return SequenceTokenizer(input_length), SequenceTokenizer(output_length)
        case _:
            raise ValueError(f"Model {model} not supported")


def preprocessing(
    project_dir: Path, splits: (float, float, float), model: Models = Models.LSTM
) -> None:
    """
    Build the input for the lstm model.
    :param project_dir: The directory created for each project.
    :param splits: Triple(train_percentile, val_percentile, test_percentile)
    :param model: The model to be used.
    :return:
    """
    violation_dir = project_dir / VIOLATION_DIR
    protocol_dirs = [
        violation_dir / Path(directory) for directory in os.listdir(violation_dir)
    ]
    for protocol_violation_dir in protocol_dirs:
        _build_splits(protocol_violation_dir, splits)
        src_tokenizer, trg_tokenizer = _build_model_tokenizers(model)
        src_vocab, trg_vocab = _build_vocab(
            protocol_violation_dir, src_tokenizer, trg_tokenizer, model
        )

        # Process train examples
        train_input_dir = (
            protocol_violation_dir
            / MODEL_DATA_PATH
            / _get_protocol_from_path(protocol_violation_dir)
            / TRAIN_PATH
        )
        train_output_dir = (
            protocol_violation_dir
            / MODEL_DATA_PATH
            / _get_protocol_from_path(protocol_violation_dir)
            / model.name.lower()
            / TRAIN_PATH
        )
        os.makedirs(train_output_dir, exist_ok=True)
        _build_inputs_from_vocab(
            train_input_dir,
            train_output_dir,
            src_vocab,
            trg_vocab,
            src_tokenizer,
            trg_tokenizer,
        )

        # Process val examples
        val_input_dir = (
            protocol_violation_dir
            / MODEL_DATA_PATH
            / _get_protocol_from_path(protocol_violation_dir)
            / VAL_PATH
        )
        val_output_dir = (
            protocol_violation_dir
            / MODEL_DATA_PATH
            / _get_protocol_from_path(protocol_violation_dir)
            / model.name.lower()
            / VAL_PATH
        )
        os.makedirs(val_output_dir, exist_ok=True)
        _build_inputs_from_vocab(
            val_input_dir,
            val_output_dir,
            src_vocab,
            trg_vocab,
            src_tokenizer,
            trg_tokenizer,
        )

        if splits[2] > 0:
            # Process test examples
            test_input_dir = (
                protocol_violation_dir
                / MODEL_DATA_PATH
                / _get_protocol_from_path(protocol_violation_dir)
                / TEST_PATH
            )
            test_output_dir = (
                protocol_violation_dir
                / MODEL_DATA_PATH
                / _get_protocol_from_path(protocol_violation_dir)
                / str(model.name).lower()
                / TEST_PATH
            )
            os.makedirs(test_output_dir, exist_ok=True)
            _build_inputs_from_vocab(
                test_input_dir,
                test_output_dir,
                src_vocab,
                trg_vocab,
                src_tokenizer,
                trg_tokenizer,
            )


def load_vocabs(src_vocab: Path, trg_vocab: Path) -> (bidict, bidict):
    """
    Load the vocabularies from the given paths.
    :param src_vocab:
    :param trg_vocab:
    :return:
    """
    src_vocab = _load_vocab_from_path(src_vocab)
    trg_vocab = _load_vocab_from_path(trg_vocab)
    return src_vocab, trg_vocab


def _load_vocab_from_path(vocab_path: Path) -> bidict[int, str]:
    """
    Load the vocabulary from the given path.
    :param vocab_path: The given path.
    :return: Return the vocabulary.
    """
    raw_json = read_content_of_file(vocab_path)
    vocab = json.loads(raw_json)
    vocab = {int(k): v for k, v in vocab.items()}
    return bidict(vocab)
