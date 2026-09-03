"""Parse and judge matched MoonEP/DeepEP Qwen3.5 acceptance runs."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import runpy
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_LOSS_PREFIX = "loss/"


@dataclass(frozen=True)
class AcceptanceRun:
    backend: str
    mtp: bool
    pack_length: int
    records: tuple[dict[str, Any], ...]

    @classmethod
    def from_tracker(
        cls,
        tracker: str | Path,
        *,
        backend: str,
        mtp: bool,
        pack_length: int,
    ) -> "AcceptanceRun":
        tracker = Path(tracker)
        records = tuple(json.loads(line) for line in tracker.read_text(encoding="utf-8").splitlines() if line)
        steps = [record.get("step") for record in records]
        if steps != list(range(1, 21)):
            raise ValueError(f"{tracker} must contain exactly steps 1..20, got {steps}")

        required = {"runtime_info/text_tokens", "runtime_info/tgs", "loss/reduced_llm_loss", "grad_norm"}
        if mtp:
            required.add("loss/reduced_mtp_loss")
        for record in records:
            missing = required - record.keys()
            if missing:
                raise ValueError(f"step {record['step']} is missing metrics: {sorted(missing)}")
        return cls(backend=backend, mtp=mtp, pack_length=pack_length, records=records)

    @classmethod
    def from_work_dir(cls, work_dir: str | Path) -> "AcceptanceRun":
        work_dir = Path(work_dir)
        manifest = json.loads((work_dir / "acceptance_manifest.json").read_text(encoding="utf-8"))
        trackers = list(work_dir.glob("**/exp_tracking/rank0/tracker.jsonl"))
        if len(trackers) != 1:
            raise ValueError(f"expected one rank0 tracker below {work_dir}, got {trackers}")
        return cls.from_tracker(
            trackers[0],
            backend=manifest["backend"],
            mtp=manifest["mtp"],
            pack_length=manifest["pack_length"],
        )

    @property
    def steps(self) -> list[int]:
        return [int(record["step"]) for record in self.records]

    @property
    def tokens(self) -> list[int]:
        return [int(record["runtime_info/text_tokens"]) for record in self.records]

    @property
    def throughput(self) -> list[float]:
        return [float(record["runtime_info/tgs"]) for record in self.records]

    def curves(self) -> dict[str, list[float]]:
        names = {
            key.removeprefix(_LOSS_PREFIX)
            for record in self.records
            for key in record
            if key.startswith(f"{_LOSS_PREFIX}reduced_") and key.endswith("loss")
        }
        curves = {name: [float(record[f"{_LOSS_PREFIX}{name}"]) for record in self.records] for name in sorted(names)}
        curves["total_loss"] = [
            sum(
                float(value)
                for key, value in record.items()
                if key.startswith(f"{_LOSS_PREFIX}reduced_") and key.endswith("loss")
            )
            for record in self.records
        ]
        curves["grad_norm"] = [float(record["grad_norm"]) for record in self.records]
        return curves


@dataclass(frozen=True)
class CurveComparison:
    cosine_similarity: float
    mean_relative_difference: float
    finite: bool
    passed: bool


@dataclass(frozen=True)
class PairComparison:
    throughput_steps: list[int]
    deepep_throughput: list[float]
    moonep_throughput: list[float]
    deepep_median: float
    moonep_median: float
    throughput_ratio: float
    curves: dict[str, CurveComparison]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compare_curve(
    reference: list[float],
    actual: list[float],
    *,
    minimum_cosine: float,
    maximum_relative_difference: float,
) -> CurveComparison:
    finite = all(math.isfinite(value) for value in [*reference, *actual])
    dot = sum(expected * observed for expected, observed in zip(reference, actual, strict=True))
    reference_norm = math.sqrt(sum(value * value for value in reference))
    actual_norm = math.sqrt(sum(value * value for value in actual))
    if reference_norm == 0 or actual_norm == 0:
        cosine = 1.0 if reference == actual else 0.0
    else:
        cosine = dot / (reference_norm * actual_norm)
    relative = statistics.fmean(
        abs(observed - expected) / max(abs(expected), 1e-12)
        for expected, observed in zip(reference, actual, strict=True)
    )
    return CurveComparison(
        cosine_similarity=cosine,
        mean_relative_difference=relative,
        finite=finite,
        passed=finite and cosine >= minimum_cosine and relative < maximum_relative_difference,
    )


def compare_runs(deepep: AcceptanceRun, moonep: AcceptanceRun) -> PairComparison:
    if deepep.backend != "deepep" or moonep.backend != "moonep":
        raise ValueError(f"expected deepep/moonep pair, got {deepep.backend}/{moonep.backend}")
    for field in ("mtp", "pack_length"):
        if getattr(deepep, field) != getattr(moonep, field):
            raise ValueError(f"workload mismatch for {field}: {getattr(deepep, field)} != {getattr(moonep, field)}")
    if deepep.tokens != moonep.tokens:
        raise ValueError("workload mismatch for per-step text tokens")

    throughput_slice = slice(5, 20)
    deepep_throughput = deepep.throughput[throughput_slice]
    moonep_throughput = moonep.throughput[throughput_slice]
    deepep_median = statistics.median(deepep_throughput)
    moonep_median = statistics.median(moonep_throughput)
    throughput_ratio = moonep_median / deepep_median

    deepep_curves = deepep.curves()
    moonep_curves = moonep.curves()
    if deepep_curves.keys() != moonep_curves.keys():
        raise ValueError(f"metric mismatch: deepep={sorted(deepep_curves)}, moonep={sorted(moonep_curves)}")
    curves = {
        name: _compare_curve(
            deepep_curves[name],
            moonep_curves[name],
            minimum_cosine=0.98 if name == "grad_norm" else 0.99,
            maximum_relative_difference=0.05 if name == "grad_norm" else 0.03,
        )
        for name in deepep_curves
    }
    return PairComparison(
        throughput_steps=list(range(6, 21)),
        deepep_throughput=deepep_throughput,
        moonep_throughput=moonep_throughput,
        deepep_median=deepep_median,
        moonep_median=moonep_median,
        throughput_ratio=throughput_ratio,
        curves=curves,
        passed=throughput_ratio >= 0.95 and all(curve.passed for curve in curves.values()),
    )


def _git_commit(directory: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()


def capture_manifest(config_path: Path, output: Path) -> None:
    trainer = runpy.run_path(str(config_path))["trainer"]
    moonep = importlib.import_module("moonep")
    torch = importlib.import_module("torch")
    moonep_source = Path(moonep.__file__).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    payload = {
        "backend": os.environ["MOONEP_ACCEPTANCE_BACKEND"],
        "mtp": bool(int(os.environ["MOONEP_ACCEPTANCE_MTP"])),
        "pack_length": int(os.environ["MOONEP_ACCEPTANCE_PACK_LENGTH"]),
        "xtuner_commit": _git_commit(repo_root),
        "moonep_commit": _git_commit(moonep_source.parents[1]),
        "moonep_module": str(moonep_source),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "configuration": json.loads(trainer.model_dump_json(serialize_as_any=True)),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "MODEL_COMPILE",
                "XTUNER_DETERMINISTIC",
                "XTUNER_ACTIVATION_OFFLOAD",
                "XTUNER_USE_CUTLASS_GROUP_GEMM",
                "XTUNER_COMPILE_NO_INPLACE_BUFFERS",
            )
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--deepep", type=Path, required=True)
    compare.add_argument("--moonep", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "capture":
        capture_manifest(args.config, args.output)
        return 0

    result = compare_runs(AcceptanceRun.from_work_dir(args.deepep), AcceptanceRun.from_work_dir(args.moonep))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
