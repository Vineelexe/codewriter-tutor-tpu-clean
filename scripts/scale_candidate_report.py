from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.config import load_model_config  # noqa: E402
from cw360.model.count import build_parameter_report  # noqa: E402
from cw360.train.validate import validate_model_architecture  # noqa: E402

DEFAULT_RECOMMENDATION = (
    "First prove TPU pipeline on 354M or tiny/354M smoke.\n"
    "Then choose between 420M and 480M based on measured TPU tokens/sec, loss curve, "
    "checkpoint size, and local inference feasibility."
)


@dataclass(frozen=True, slots=True)
class ScaleCandidate:
    config_path: str
    size_label: str
    parameter_count: int
    expected_parameter_count: int | None
    fp32_parameter_gb: float
    bf16_parameter_gb: float
    int8_parameter_gb: float
    rough_training_checkpoint_gb: float
    local_inference_burden: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScaleCandidateReport:
    candidates: tuple[ScaleCandidate, ...]
    recommended_initial_tpu_pilot: str
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "recommended_initial_tpu_pilot": self.recommended_initial_tpu_pilot,
            "recommendation": self.recommendation,
        }


def build_scale_candidate_report(config_paths: list[str | Path]) -> ScaleCandidateReport:
    candidates = tuple(_candidate_from_config(Path(path)) for path in config_paths)
    return ScaleCandidateReport(
        candidates=candidates,
        recommended_initial_tpu_pilot="354m",
        recommendation=DEFAULT_RECOMMENDATION,
    )


def _candidate_from_config(path: Path) -> ScaleCandidate:
    config = load_model_config(path)
    validate_model_architecture(config, require_explicit_mlp_hidden_size=True)
    report = build_parameter_report(config)
    parameter_count = report.total_parameters
    return ScaleCandidate(
        config_path=str(path),
        size_label=config.size_label,
        parameter_count=parameter_count,
        expected_parameter_count=report.expected_parameters,
        fp32_parameter_gb=_gb(parameter_count * 4),
        bf16_parameter_gb=_gb(parameter_count * 2),
        int8_parameter_gb=_gb(parameter_count),
        rough_training_checkpoint_gb=_gb(parameter_count * 12),
        local_inference_burden=_local_inference_burden(config.size_label),
    )


def _gb(num_bytes: int) -> float:
    return round(num_bytes / (1024**3), 3)


def _local_inference_burden(size_label: str) -> str:
    if size_label == "354m":
        return "lowest candidate burden; best first full-scale CPU inference check"
    if size_label == "420m":
        return "middle candidate burden; compare after TPU throughput is measured"
    if size_label == "480m":
        return "highest candidate burden; only choose if TPU and local inference checks justify it"
    return "tiny smoke only"


def format_report(report: ScaleCandidateReport) -> str:
    lines = [
        "scale candidate report",
        "size_label params fp32_gb bf16_gb int8_gb rough_training_checkpoint_gb local_burden",
    ]
    for candidate in report.candidates:
        lines.append(
            f"{candidate.size_label} {candidate.parameter_count} "
            f"{candidate.fp32_parameter_gb:.3f} {candidate.bf16_parameter_gb:.3f} "
            f"{candidate.int8_parameter_gb:.3f} "
            f"{candidate.rough_training_checkpoint_gb:.3f} "
            f"{candidate.local_inference_burden}"
        )
    lines.append("")
    lines.append(report.recommendation)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TPU scale candidate model configs.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_scale_candidate_report(args.configs)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()
