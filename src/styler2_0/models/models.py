import json
from enum import Enum
from pathlib import Path

from bidict import bidict
from streamerate import stream
from torch import Tensor, long, nn, tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.styler2_0.models.lstm import LSTM
from src.styler2_0.models.model_base import ModelBase
from src.styler2_0.utils.checkstyle import run_checkstyle_on_str
from src.styler2_0.utils.java import is_parseable
from src.styler2_0.utils.tokenize import ProcessedSourceFile, tokenize_java_code
from src.styler2_0.utils.utils import (
    get_files_in_dir,
    get_sub_dirs_in_dir,
    read_content_of_file,
)
from src.styler2_0.utils.vocab import Vocabulary

TRAIN_DATA = Path("train")
TRAIN_SRC = TRAIN_DATA / Path("input.txt")
TRAIN_TRG = TRAIN_DATA / Path("ground_truth.txt")
VAL_DATA = Path("val")
VAL_SRC = VAL_DATA / Path("input.txt")
VAL_TRG = VAL_DATA / Path("ground_truth.txt")
SRC_VOCAB_FILE = Path("src_vocab.txt")
TRG_VOCAB_FILE = Path("trg_vocab.txt")
CHECKPOINT_FOLDER = Path("checkpoints")
BEST_MODEL = Path("best_model.pt")
META_DATA_EVAL = Path("data.json")


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
    project_dir: Path, src_vocab: Vocabulary, trg_vocab: Vocabulary, model: ModelBase
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
        DataLoader(list(zip(train_inp, train_trg, strict=True)), batch_size=32),
        DataLoader(list(zip(val_inp, val_trg, strict=True)), batch_size=32),
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


def _line_to_tensor(line: str, dim: int, vocab: Vocabulary) -> list:
    """
    Load the given line to a tensor.
    :param line: The given line.
    :param dim: The dimension of the tensor.
    :param vocab: The vocab.
    :return: The tensor.
    """
    raw_tensor = list(stream(line.split(" ")).map(int).to_list())
    raw_tensor.extend([vocab[vocab.pad]] * (dim - len(raw_tensor)))
    return raw_tensor


def _load_vocabs(project_dir: Path) -> tuple[Vocabulary, Vocabulary]:
    """
    Load the vocabs from the given project dir.
    :param project_dir: The given project dir.
    :return: The vocabs.
    """
    return Vocabulary.load(project_dir / SRC_VOCAB_FILE), Vocabulary.load(
        project_dir / TRG_VOCAB_FILE
    )


def train(model_type: Models, model_dir: Path, epochs: int) -> None:
    """
    Train the given model on each protocol separately.
    :param model_type: The given model.
    :param model_dir: Project dir with the data.
    :param epochs: Count epochs.
    :return:
    """

    # TODO:
    #    - Batch sizes as cl args
    #    - Criterion and optimizer from config
    protocol_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    for model_protocol_dir in protocol_dirs:
        src_vocab, trg_vocab = _load_vocabs(model_protocol_dir)

        model = model_type.value.build_from_config(
            src_vocab, trg_vocab, model_protocol_dir / CHECKPOINT_FOLDER
        )
        train_data, val_data = _load_train_and_val_data(
            model_protocol_dir, src_vocab, trg_vocab, model
        )
        criterion = nn.CrossEntropyLoss(ignore_index=src_vocab[src_vocab.pad])
        optimizer = Adam(model.parameters())
        model.fit(epochs, train_data, val_data, criterion, optimizer)


def evaluate(
    model: Models,
    mined_violations_dir: Path,
    model_data_dirs: list[Path],
    checkpoints: list[Path],
    top_k: int = 5,
) -> None:
    """
    Evaluate the given models (all the same type) combined on the mined violations
    (Same as styler). Mined that checkpoints and model_data_dirs are in the same order.
    :param model:
    :param mined_violations_dir:
    :param model_data_dirs:
    :param checkpoints:
    :param top_k:
    :return:
    """
    assert len(model_data_dirs) == len(
        checkpoints
    ), "The number of model data dirs and checkpoints must be the same."
    vocabs: list[(Vocabulary, Vocabulary)] = [_load_vocabs(d) for d in model_data_dirs]
    models: list[ModelBase] = []
    for index, checkpoint in enumerate(checkpoints):
        src_vocab, trg_vocab = vocabs[index]
        model = model.value.load_from_config(src_vocab, trg_vocab, checkpoint)
        models.append(model)
    meta_data = json.loads(read_content_of_file(mined_violations_dir / META_DATA_EVAL))
    config = meta_data["config"]
    version = meta_data["version"]
    all_violation_dirs = get_sub_dirs_in_dir(mined_violations_dir)
    fixed = 0
    not_fixed = 0
    for violation in tqdm(all_violation_dirs, "Evaluating fixing of violations"):
        violation_file = get_files_in_dir(violation / "violation")[0]
        violation_content = read_content_of_file(violation_file)
        report = run_checkstyle_on_str(violation_content, version, config)
        tokens = tokenize_java_code(violation_content)
        processed_file = ProcessedSourceFile(None, tokens, report)
        possible_fixes = []
        for model in models:
            possible_fixes.extend(model.fix(processed_file, top_k))
        real_fixes = []
        for possible_fix in possible_fixes:
            if (
                is_parseable(possible_fix.de_tokenize())
                and not run_checkstyle_on_str(
                    possible_fix.de_tokenize(), version, config
                ).violations
            ):
                real_fixes.append(possible_fix)
        if real_fixes:
            print("fixed")
            fixed += 1
        else:
            print("not fixed")
            not_fixed += 1
    print(f"Fixed: {fixed}, Not fixed: {not_fixed}")
