import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import groupby
from pathlib import Path

from bidict import bidict
from streamerate import stream
from torch import Tensor, long, nn, tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.styler2_0.models.lstm import LSTM
from src.styler2_0.models.model_base import ModelBase
from src.styler2_0.models.transformer import Transformer
from src.styler2_0.preprocessing.violation_generation import Protocol
from src.styler2_0.utils.checkstyle import run_checkstyle_on_dir, run_checkstyle_on_str
from src.styler2_0.utils.java import is_parseable
from src.styler2_0.utils.tokenize import ProcessedSourceFile, tokenize_java_code
from src.styler2_0.utils.utils import (
    get_files_in_dir,
    get_sub_dirs_in_dir,
    read_content_of_file,
    save_content_to_file,
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
MODEL_DATA_DIR = Path("model_data")
META_DATA_EVAL = Path("data.json")
EVAL_DATA_PATH = Path("eval_data")


class Models(Enum):
    """
    The supported models.
    """

    LSTM = LSTM
    ANN = None
    NGRAM = None
    TRANSFORMER = Transformer


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
        model_protocol_dir = model_protocol_dir / model_type.name.lower()
        src_vocab, trg_vocab = _load_vocabs(model_protocol_dir)

        model = model_type.value.build_from_config(
            src_vocab, trg_vocab, model_protocol_dir / CHECKPOINT_FOLDER
        )
        model.to(model.device)
        train_data, val_data = _load_train_and_val_data(
            model_protocol_dir, src_vocab, trg_vocab, model
        )
        criterion = nn.CrossEntropyLoss(ignore_index=src_vocab[src_vocab.pad])
        optimizer = Adam(model.parameters())
        model.fit(epochs, train_data, val_data, criterion, optimizer)


def evaluate(
    model_type: Models,
    mined_violations_dir: Path,
    project_dir: Path,
    top_k: int = 5,
) -> None:
    """
    Evaluate the given model combined (all protocols) on the mined violations
    (Same as styler).
    :param mined_violations_dir: The directory of the mined violations.
    :param model_type: The model type to be evaluated.
    :param project_dir: The processed project dir.
    :param top_k: The param for the top k accuracy.
    :return:
    """
    eval_data_dir = project_dir / EVAL_DATA_PATH
    os.makedirs(eval_data_dir, exist_ok=True)

    protocols = [p.name.lower() for p in Protocol]
    meta_data = json.loads(read_content_of_file(mined_violations_dir / META_DATA_EVAL))
    config = meta_data["config"]
    version = meta_data["version"]

    all_violation_dirs = get_sub_dirs_in_dir(mined_violations_dir)
    fix_stats = []
    for protocol in protocols:
        model_data_dir = (
            project_dir / MODEL_DATA_DIR / protocol / model_type.name.lower()
        )
        src_vocab, trg_vocab = _load_vocabs(model_data_dir)
        model = model_type.value.load_from_config(
            src_vocab,
            trg_vocab,
            model_data_dir / CHECKPOINT_FOLDER / f"{model_type.name.lower()}.pt",
        )

        for violation in tqdm(
            all_violation_dirs,
            f"Evaluating fixing of violations {model_type.name} trained on {protocol}",
        ):
            # Run checkstyle on the violation dir in order to set the path in the
            # report. This is needed to later identify the fixed violations across
            # protocols. As there is only one file in the violation dir, we can
            # take the first report.
            violation_dir = violation / "violation"
            report = next(iter(run_checkstyle_on_dir(violation_dir, version, config)))
            violation_type = next(iter(report.violations)).type
            violated_file = str(report.path)

            # Process the file to be fed to the model.
            violation_content = read_content_of_file(get_files_in_dir(violation_dir)[0])
            tokens = tokenize_java_code(violation_content)
            processed_file = ProcessedSourceFile(None, tokens, report)

            # Get the possible fixes, by calling fix on the model.
            possible_fixes = model.fix(processed_file, top_k)

            # Check if the fixes are valid.
            real_fixes = []
            for possible_fix in possible_fixes:
                # Check if fix compiles and passes checkstyle without violations.
                if (
                    is_parseable(possible_fix.de_tokenize())
                    and not run_checkstyle_on_str(
                        possible_fix.de_tokenize(), version, config
                    ).violations
                ):
                    real_fixes.append(possible_fix)
            if real_fixes:
                shortest = min(real_fixes, key=lambda x: len(x.de_tokenize()))
                fix_stat = FixStats(
                    violated_file,
                    violation_type.value,
                    protocol,
                    True,
                    len(shortest.de_tokenize()),
                )
            else:
                fix_stat = FixStats(
                    violated_file, violation_type.value, protocol, False
                )
            fix_stats.append(fix_stat)
    eval_stats = EvalStats(fix_stats)
    save_content_to_file(
        eval_data_dir / f"{model_type.name.lower()}_eval_stats.json",
        eval_stats.to_json(),
    )


@dataclass(eq=True, frozen=True)
class FixStats:
    """
    The stats of a fix.
    """

    violated_file: str
    violation_type: str
    protocol: str
    fixed: bool
    len_of_fix: int = field(default=-1)


class EvalStats:
    """
    The evaluation stats.
    """

    def __init__(self, fix_stats: list[FixStats]):
        """
        Init eval stats.
        :param fix_stats: The fix stats collected from the evaluation.
        """
        self.grouped_by_violated_file = {
            k: list(vs) for k, vs in groupby(fix_stats, lambda s: s.violated_file)
        }
        self.protocols = {s.protocol for s in fix_stats}
        self.all_violation_types = {s.violation_type for s in fix_stats}

    def _fixed_by_any_model(self) -> list[str]:
        """
        Returns the files that were fixed by any model.
        :return: The files that were fixed by any model.
        """
        return [
            file
            for file, stats in self.grouped_by_violated_file.items()
            if any(s.fixed for s in stats)
        ]

    def macro_acc(self) -> float:
        """
        Returns the macro accuracy.
        :return: The macro accuracy.
        """
        return len(self._fixed_by_any_model()) / len(self.grouped_by_violated_file)

    def micro_acc(self) -> dict[str, float]:
        """
        Returns the micro accuracy per protocol.
        :return: The micro accuracy per protocol.
        """
        return {
            protocol: len(self._fixed_by_protocol(protocol))
            / len(self.grouped_by_violated_file)
            for protocol in self.protocols
        }

    def _fixed_by_protocol(self, protocol: str) -> list[str]:
        """
        Returns the files that were fixed by the given protocol.
        :param protocol: The protocol.
        :return: The files that were fixed by the given protocol.
        """
        return list(
            stream(self.grouped_by_violated_file.items())
            .flatMap(lambda x: stream(x[1]).map(lambda s: (x[0], s)))
            .filter(lambda x: x[1].protocol == protocol and x[1].fixed)
            .map(lambda f, _: f)
            .to_list()
        )

    def to_json(self) -> str:
        """
        Returns the json representation of the eval stats.
        :return: The json representation of the eval stats.
        """
        return json.dumps(
            {
                "macro_acc": self.macro_acc(),
                "micro_acc": self.micro_acc(),
                "stats": {
                    p: [asdict(s) for s in ss]
                    for p, ss in self.grouped_by_violated_file.items()
                },
            }
        )
