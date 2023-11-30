import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import time

from streamerate import stream
from tqdm import tqdm

from styler2_0.utils.utils import (
    get_files_in_dir,
    get_sub_dirs_in_dir,
    read_content_of_file,
    save_content_to_file,
)


@dataclass(frozen=True, eq=True)
class EpochStats:
    """
    Data class for epoch stats.
    """

    epoch: int
    train_loss: float
    valid_loss: float

    def to_json(self) -> str:
        """
        Convert to json.
        :return: Returns the json string.
        """
        return json.dumps(asdict(self))


@dataclass(eq=True)
class TrainStats:
    """
    Data class for training stats.
    """

    best_epoch: int
    epoch_stats: list[EpochStats]
    start_time: int = int(time())
    end_time: int = int(time())

    def to_json(self) -> str:
        """
        Convert to json.
        :return: Returns the json string.
        """

        # Update end time every time when stats are saved
        self.end_time = int(time())
        return json.dumps(asdict(self))


@dataclass(frozen=True)
class FixStats:
    """
    The stats of a fix.
    """

    violated_file: str
    violation_type: str
    protocol: str
    fixed: bool
    len_of_fix: int = field(default=-1)

    @classmethod
    def from_json(cls, json_str: str) -> "FixStats":
        """
        Creates a FixStats object from json.
        :param json_str: The json string.
        :return: The FixStats object.
        """
        return FixStats(**json.loads(json_str))


class EvalStatsPerModel:
    """
    The evaluation stats.
    """

    def __init__(self, fix_stats: list[FixStats]):
        """
        Init eval stats.
        :param fix_stats: The fix stats collected from the evaluation.
        """
        self.grouped_by_violated_file = defaultdict(list)
        for fix_stat in fix_stats:
            self.grouped_by_violated_file[fix_stat.violated_file].append(fix_stat)
        self.grouped_by_violation_type = defaultdict(list)
        for file, stats in self.grouped_by_violated_file.items():
            self.grouped_by_violation_type[stats[0].violation_type].append(
                {file: stats}
            )
        self.protocols = {s.protocol for s in fix_stats}

    def fixed_by_any_model(self) -> list[str]:
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
        return len(self.fixed_by_any_model()) / len(self.grouped_by_violated_file)

    def micro_acc(self) -> dict[str, float]:
        """
        Returns the micro accuracy per protocol.
        :return: The micro accuracy per protocol.
        """
        return {
            protocol: len(self.fixed_by_protocol(protocol))
            / len(self.grouped_by_violated_file)
            for protocol in self.protocols
        }

    def fixed_by_protocol(self, protocol: str) -> list[str]:
        """
        Returns the files that were fixed by the given protocol.
        :param protocol: The protocol.
        :return: The files that were fixed by the given protocol.
        """
        return list(
            stream(self.grouped_by_violated_file.items())
            .flatMap(lambda x: stream(x[1]).map(lambda s: (x[0], s)))
            .filter(lambda x: x[1].protocol == protocol and x[1].fixed)
            .map(lambda x: x[0])
            .to_list()
        )

    def micro_acc_by_violation_type(self) -> dict[str, float]:
        """
        Calculates the micro accuracy of fixing each violation individually.
        :return: Returns the accuracy for each seen violation type.
        """
        return {
            vio_type: len(self.fixed_by_violation_type(vio_type)) / len(stats)
            for vio_type, stats in self.grouped_by_violation_type.items()
        }

    def fixed_by_violation_type(self, violation_type: str) -> list[str]:
        return list(
            stream(self.grouped_by_violation_type[violation_type])
            .map(lambda fs: stream(fs.values()).flatMap())
            .filter(lambda fs: any(f.fixed for f in fs))
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
                "stats_per_type": self.micro_acc_by_violation_type(),
                "stats": {
                    p: [asdict(s) for s in ss]
                    for p, ss in self.grouped_by_violated_file.items()
                },
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "EvalStatsPerModel":
        """
        Creates an EvalStats object from json.
        :param json_str: The json string.
        :return: The EvalStats object.
        """
        return EvalStatsPerModel(
            [
                FixStats.from_json(json.dumps(s))
                for s in json.loads(json_str)["stats"].values()
            ]
        )


class EvalStats:
    """
    The evaluation stats over all models.
    """

    def __init__(self, stats_per_models: dict[str, EvalStatsPerModel]) -> None:
        self.stats_per_models = stats_per_models
        self.macro_acc = self.calculate_macro_over_all_models()
        self.micro_acc = self.calculate_protocols_micro_over_all_models()
        self.type_micro = self.calculate_type_micro_over_all_models()

    def calculate_macro_over_all_models(self) -> float:
        """
        Calculates the macro accuracy over all models.
        :return: The macro accuracy over all models.
        """
        fixed_violation = set()
        for stat_per_model in self.stats_per_models.values():
            fixed_violation.update(stat_per_model.fixed_by_any_model())
        return len(fixed_violation) / len(
            next(iter(self.stats_per_models.values())).grouped_by_violated_file
        )

    def calculate_protocols_micro_over_all_models(self) -> dict[str, float]:
        """
        Calculates the micro accuracy over all models.
        :return: The micro accuracy over all models.
        """
        first_stats = next(iter(self.stats_per_models.values()))
        protocols = first_stats.protocols
        amount_violations = len(first_stats.grouped_by_violated_file)
        fixes_per_protocol = defaultdict(float)
        for protocol in protocols:
            fixed_violation = set()
            for stat_per_model in self.stats_per_models.values():
                fixed_violation.update(stat_per_model.fixed_by_protocol(protocol))
            fixes_per_protocol[protocol] = len(fixed_violation) / amount_violations
        return fixes_per_protocol

    def calculate_type_micro_over_all_models(self) -> dict[str, float]:
        """
        Calculates the micro accuracy over all models.
        :return: The micro accuracy over all models.
        """
        first_stats = next(iter(self.stats_per_models.values()))
        amount_violations = len(first_stats.grouped_by_violated_file)
        fixes_per_type = defaultdict(float)
        for violation_type in first_stats.grouped_by_violation_type:
            fixed_violation = set()
            for stat_per_model in self.stats_per_models.values():
                fixed_violation.update(
                    stat_per_model.fixed_by_violation_type(violation_type)
                )
            fixes_per_type[violation_type] = len(fixed_violation) / amount_violations
        return fixes_per_type

    def to_json(self) -> str:
        """
        Returns the json representation of the eval stats.
        :return: The json representation of the eval stats.
        """
        return json.dumps(
            {
                "macro_acc": self.macro_acc,
                "micro_acc": self.micro_acc,
                "stats_per_type": self.type_micro,
            },
            indent=2,
        )


def analyze_mined_violations(mined_violations_dir: Path) -> None:
    """
    Analyze the mined violations.
    :param mined_violations_dir: The directory with the mined violations.
    :return:
    """
    violation_dirs = get_sub_dirs_in_dir(mined_violations_dir)
    violation_amount_dict = defaultdict(int)
    violation_fixed_dict = defaultdict(int)
    for violation in tqdm(violation_dirs, desc="Analyzing mined violations"):
        violation_data = json.loads(read_content_of_file(violation / "data.json"))
        violation_amount_dict[violation_data["violation_type"]] += 1
        if violation_data["fixed"]:
            violation_fixed_dict[violation_data["violation_type"]] += 1

    violation_amount = sum(violation_amount_dict.values())
    violation_fixed = sum(violation_fixed_dict.values())
    mined_violations_data = {
        "violation_amount": violation_amount,
        "violation_fixed": violation_fixed,
        "violation_amount_dict": dict(violation_amount_dict),
        "violation_fixed_dict": dict(violation_fixed_dict),
    }
    save_content_to_file(
        mined_violations_dir / "analysis.json", json.dumps(mined_violations_data)
    )


def analyze_all_eval_jsons(eval_dir: Path) -> None:
    """
    Analyze all eval jsons.
    :param eval_dir: The directory with the eval jsons.
    :return:
    """
    eval_jsons = get_files_in_dir(eval_dir, ".json")
    eval_json_per_model = {
        e.name.split("_")[0]: EvalStatsPerModel.from_json(read_content_of_file(e))
        for e in eval_jsons
    }
    eval_stats = EvalStats(eval_json_per_model)
    save_content_to_file(eval_dir / "eval_data.json", eval_stats.to_json())
