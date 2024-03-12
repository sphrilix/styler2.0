import json
import os
import random
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
        fixed_by = []
        for fs in self.grouped_by_violation_type[violation_type]:
            for file, fix_stats in fs.items():
                if any(fix_stat.fixed for fix_stat in fix_stats):
                    fixed_by.append(file)
        return fixed_by

    def to_json(self) -> str:
        """
        Returns the json representation of the eval stats.
        :return: The json representation of the eval stats.
        """
        return json.dumps(
            self.as_dict(),
            indent=2,
        )

    def as_dict(self) -> dict[str, ...]:
        return {
            "macro_acc": self.macro_acc(),
            "micro_acc": self.micro_acc(),
            "stats_per_type": self.micro_acc_by_violation_type(),
            "stats": {
                p: [asdict(s) for s in ss]
                for p, ss in self.grouped_by_violated_file.items()
            },
        }

    @classmethod
    def from_json(cls, json_str: str) -> "EvalStatsPerModel":
        """
        Creates an EvalStats object from json.
        :param json_str: The json string.
        :return: The EvalStats object.
        """
        return EvalStatsPerModel(
            [
                FixStats.from_json(json.dumps(f))
                for fs in json.loads(json_str)["stats"].values()
                for f in fs
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
        fixes_per_type = defaultdict(float)
        for violation_type in first_stats.grouped_by_violation_type:
            amount_violations = len(
                first_stats.grouped_by_violation_type[violation_type]
            )
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
                "stats_per_model": {
                    model: stats.as_dict()
                    for model, stats in self.stats_per_models.items()
                },
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "EvalStats":
        """
        Creates an EvalStats object from json.
        :param json_str: The json string.
        :return: The EvalStats object.
        """
        return EvalStats(
            {
                model: EvalStatsPerModel.from_json(json.dumps(stats))
                for model, stats in json.loads(json_str)["stats_per_model"].items()
            }
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
        if e.name != "eval_data.json"
    }
    eval_stats = EvalStats(eval_json_per_model)
    save_content_to_file(eval_dir / "eval_data.json", eval_stats.to_json())


def analyze_generated_violations(violation_dir: Path) -> None:
    """
    Analyze the generated violations.
    :param violation_dir: The directory with the generated violations.
    :return:
    """
    generated_statistics = {}
    protocols = get_sub_dirs_in_dir(violation_dir)
    for protocol in protocols:
        violations = get_sub_dirs_in_dir(protocol)
        statistics_per_protocol = {
            "violations": len(violations),
            "violations_per_type": defaultdict(int),
        }
        for violation in tqdm(
            violations, desc=f"Analyzing generated violations of {protocol.name}"
        ):
            if not Path(violation / "data.json").exists():
                continue
            data_json = json.loads(read_content_of_file(violation / "data.json"))
            statistics_per_protocol["violations_per_type"][
                data_json["violation_type"]
            ] += 1

        generated_statistics[protocol.name] = statistics_per_protocol
    save_content_to_file(
        violation_dir / "analysis.json", json.dumps(generated_statistics, indent=2)
    )


def analyze_data_dir(projects_dir: Path) -> None:
    """
    Analyze all eval data.
    :param projects_dir: The directory with the projects.
    :return:
    """
    all_mined_vios = defaultdict(dict)
    all_eval_datas = {}
    projects = [d for d in get_sub_dirs_in_dir(projects_dir) if d.name.endswith("out")]
    print(projects)
    for project in tqdm(projects, desc="Loading meta data:"):
        eval_dir = project / "eval_data"
        mined_violations_dir = project / "mined_violations"
        try:
            mined_vio_data = json.loads(
                read_content_of_file(mined_violations_dir / "analysis.json")
            )
            analyze_all_eval_jsons(eval_dir)
            eval_data = EvalStats.from_json(
                read_content_of_file(eval_dir / "eval_data.json")
            )
        except Exception:
            continue
        all_mined_vios[project.name] = mined_vio_data

        all_eval_datas[project.name] = eval_data
    save_content_to_file(
        projects_dir / "eval.json",
        json.dumps(_process_all_fix_stats(all_eval_datas, all_mined_vios), indent=2),
    )


def _process_all_mined_violations(mined_data: dict[str, ...]) -> dict[str, ...]:
    amount_vio = 0
    amount_vio_per_type = defaultdict(int)
    for vio_data in mined_data.values():
        amount_vio += vio_data["violation_amount"]
        for vio_type, vio_amount in vio_data["violation_amount_dict"].items():
            amount_vio_per_type[vio_type] += vio_amount
    return {
        "all_violation_amount": amount_vio,
        "all_violation_amount_dict": amount_vio_per_type,
    }


def _process_all_fix_stats(
    all_eval_datas: dict[str, ...], all_mined_vios: dict[str, ...]
) -> dict[str, ...]:
    combined_mined_vios = _process_all_mined_violations(all_mined_vios)
    micro_over_all_models_per_protocol = defaultdict(float)
    micro_over_all_models_per_type = defaultdict(float)
    acc_per_model = {}
    macro_acc = 0.0
    all_stats_per_type = _all_stats_per_type(all_eval_datas)
    normalized_acc = _normalized_acc_per_type(all_stats_per_type)
    for project, eval_data in all_eval_datas.items():
        try:
            project_mined_vios = all_mined_vios[project]
            macro_acc += (
                eval_data.macro_acc * project_mined_vios["violation_amount"]
            ) / combined_mined_vios["all_violation_amount"]
            for prot, acc in eval_data.micro_acc.items():
                micro_over_all_models_per_protocol[prot] += (
                    acc * project_mined_vios["violation_amount"]
                ) / combined_mined_vios["all_violation_amount"]
            for vio, acc in eval_data.type_micro.items():
                micro_over_all_models_per_type[vio] += (
                    acc * project_mined_vios["violation_amount_dict"][vio]
                ) / combined_mined_vios["all_violation_amount_dict"][vio]
            for model, stats in eval_data.stats_per_models.items():
                if not acc_per_model.get(model):
                    acc_per_model[model] = {"acc": 0.0, "acc_per_type": {}}
                acc_per_model[model]["acc"] += (
                    stats.macro_acc() * project_mined_vios["violation_amount"]
                ) / combined_mined_vios["all_violation_amount"]
                for vio_type, vio_acc in stats.micro_acc_by_violation_type().items():
                    if not acc_per_model[model]["acc_per_type"].get(vio_type):
                        acc_per_model[model]["acc_per_type"][vio_type] = 0.0
                    acc_per_model[model]["acc_per_type"][vio_type] += (
                        vio_acc * project_mined_vios["violation_amount_dict"][vio_type]
                    ) / combined_mined_vios["all_violation_amount_dict"][vio_type]
        except Exception:
            continue

    return {
        "acc": macro_acc,
        "acc_per_protocol": micro_over_all_models_per_protocol,
        "acc_per_type": micro_over_all_models_per_type,
        "acc_per_model": acc_per_model,
        "normalized_acc": normalized_acc,
        "violations": combined_mined_vios,
    }


def _normalized_acc_per_type(all_stats_per_type: dict) -> dict:
    normalized = defaultdict(dict)
    minimal_count = 10
    for _ in range(1000):
        sampled_violations = _sample_vios(all_stats_per_type, minimal_count)
        _calc_norm_acc_per_model(
            all_stats_per_type, minimal_count, normalized, sampled_violations
        )

        _calc_norm_acc_over_all(
            all_stats_per_type, minimal_count, normalized, sampled_violations
        )

    normalized["combined"]["acc"] = sum(
        normalized["combined"]["acc_per_type"].values()
    ) / len(normalized["combined"]["acc_per_type"])

    return normalized


def _calc_norm_acc_per_model(
    all_stats_per_type: dict,
    minimal_count: int,
    normalized: dict,
    sampled_violations: dict,
) -> None:
    for model, stats_per_type in all_stats_per_type.items():
        normalized_per_model = defaultdict(float)
        for vio_type, files in sampled_violations.items():
            sampled = [
                next(iter(vio.values()))
                for vio in stats_per_type[vio_type]
                if next(iter(vio.keys())) in files
            ]
            sampled_fixed = len([fs for fs in sampled if any(s.fixed for s in fs)])
            normalized_per_model[vio_type] += sampled_fixed / minimal_count
        previous_acc = (
            0.0 if not normalized[model].get("acc") else normalized[model]["acc"]
        )
        previous_per_type = (
            normalized[model]["acc_per_type"]
            if normalized[model].get("acc_per_type")
            else {vt: 0.0 for vt in sampled_violations}
        )
        normalized[model] = {
            "acc": previous_acc
            + sum(normalized_per_model.values()) / len(normalized_per_model) / 1000,
            "acc_per_type": {
                vt: acc + normalized_per_model[vt] / 1000
                for vt, acc in previous_per_type.items()
            },
        }


def _calc_norm_acc_over_all(
    all_stats_per_type: dict,
    minimal_count: int,
    normalized: dict,
    sampled_violations: dict,
) -> None:
    fixed_over_models = defaultdict(set)
    for vio_type, vios in sampled_violations.items():
        for _, stats_per_type in all_stats_per_type.items():
            fixed_per_type = set()
            for vio in vios:
                curr_vio = next(
                    fs
                    for fs in stats_per_type[vio_type]
                    if next(iter(fs.keys())) == vio
                )
                if any(curr.fixed for curr in next(iter(curr_vio.values()))):
                    fixed_per_type.add(next(iter(curr_vio.keys())))

            fixed_over_models[vio_type].update(fixed_per_type)
    for vio_type, fixed in fixed_over_models.items():
        if not normalized["combined"].get("acc_per_type"):
            normalized["combined"]["acc_per_type"] = {}
        if not normalized["combined"]["acc_per_type"].get(vio_type):
            normalized["combined"]["acc_per_type"][vio_type] = 0.0
        normalized["combined"]["acc_per_type"][vio_type] += (
            len(fixed) / minimal_count / 1000
        )


def _sample_vios(all_stats_per_type: dict, min_amount: int) -> dict:
    sampled_violations = {}
    for vio_type, vios in next(iter(all_stats_per_type.values())).items():
        if len(vios) < min_amount:
            continue
        vios_per_type = random.choices(
            [next(iter(vio.keys())) for vio in vios], k=min_amount
        )
        sampled_violations[vio_type] = vios_per_type
    return sampled_violations


def _all_stats_per_type(all_eval_datas: dict) -> dict:
    all_stats_per_type = defaultdict(dict)
    for ps in all_eval_datas.values():
        for model, stats in ps.stats_per_models.items():
            for vio_type, stats_per_type in stats.grouped_by_violation_type.items():
                if not all_stats_per_type.get(model):
                    all_stats_per_type[model] = {}
                if not all_stats_per_type[model].get(vio_type):
                    all_stats_per_type[model][vio_type] = []
                all_stats_per_type[model][vio_type].extend(stats_per_type)
    return all_stats_per_type


def analyze_pre_train_data(pre_train_dir: Path) -> None:
    """
    Analyze the pre-train data.
    :param pre_train_dir: The directory with the pre-train data.
    :return:
    """
    pre_train_dirs = get_sub_dirs_in_dir(pre_train_dir / "violations/pre_training")
    pre_train_amount_dict = defaultdict(int)
    changed_lines = []
    for pre_train in tqdm(pre_train_dirs, desc="Analyzing pre train data"):
        pre_train_data = json.loads(read_content_of_file(pre_train / "data.json"))
        pre_train_amount_dict[pre_train_data["violation_type"]] += 1
        changed_lines.append(
            _changed_lines(
                pre_train_data["violated_source"], pre_train_data["non_violated_source"]
            )
        )

    pre_train_amount = sum(pre_train_amount_dict.values())
    pre_train_data = {
        "pre_train_amount": pre_train_amount,
        "pre_train_amount_dict": dict(pre_train_amount_dict),
        "changed_lines_avg": sum(changed_lines) / len(changed_lines),
        "changed_lines": changed_lines,
    }

    print(pre_train_data)
    save_content_to_file(
        pre_train_dir / "analysis.json", json.dumps(pre_train_data, indent=2)
    )


def _changed_lines(violated_source: str, non_violated_source: str) -> int:
    diff = os.popen(
        f"git diff --no-index {violated_source} {non_violated_source}"
    ).read()
    diff_lines = diff.split("\n")
    diff_lines = diff_lines[5:]
    removed_lines = 0
    added_lines = 0
    changed_lines = 0
    for line in diff_lines:
        if line.startswith("-"):
            removed_lines += 1
            continue

        if line.startswith("+"):
            added_lines += 1
            continue

        if removed_lines == added_lines:
            changed_lines += removed_lines
        else:
            cl = abs(removed_lines - added_lines)
            if removed_lines > added_lines:
                changed_lines += cl + (removed_lines - cl)
            else:
                changed_lines += cl + (added_lines - cl)

        removed_lines = 0
        added_lines = 0

    return changed_lines


def analyze_changed_lines_generated(gen_vio_dir: Path) -> None:
    """
    Analyze the changed lines of the generated violations.
    :param gen_vio_dir: The directory with the generated violations.
    :return:
    """
    protocols = get_sub_dirs_in_dir(gen_vio_dir)
    changed_lines = []
    for protocol in protocols:
        for vio in get_sub_dirs_in_dir(gen_vio_dir / protocol):
            data = json.loads(read_content_of_file(vio / "data.json"))
            changed_lines.append(
                _changed_lines(data["violated_source"], data["non_violated_source"])
            )
    save_content_to_file(gen_vio_dir / "changed_lines.json", str(changed_lines))
