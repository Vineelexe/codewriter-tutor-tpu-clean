from __future__ import annotations

from pathlib import Path

from scripts.scale_candidate_report import (
    DEFAULT_RECOMMENDATION,
    build_scale_candidate_report,
    format_report,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_scale_candidate_report_compares_all_candidate_sizes_without_final_lock() -> None:
    report = build_scale_candidate_report(
        [
            CONFIG_DIR / "model_354m.yaml",
            CONFIG_DIR / "model_420m.yaml",
            CONFIG_DIR / "model_480m.yaml",
        ]
    )

    assert [candidate.size_label for candidate in report.candidates] == ["354m", "420m", "480m"]
    assert [candidate.parameter_count for candidate in report.candidates] == sorted(
        candidate.parameter_count for candidate in report.candidates
    )
    assert {candidate.size_label: candidate.parameter_count for candidate in report.candidates} == {
        "354m": 354_417_920,
        "420m": 423_258_880,
        "480m": 478_331_648,
    }
    assert {
        candidate.size_label: candidate.expected_parameter_count for candidate in report.candidates
    } == {
        "354m": 354_000_000,
        "420m": 423_000_000,
        "480m": 479_000_000,
    }
    assert report.recommended_initial_tpu_pilot == "354m"
    assert report.recommendation == DEFAULT_RECOMMENDATION
    assert "choose between 420M and 480M" in report.recommendation


def test_scale_candidate_report_includes_memory_estimates_and_checkpoint_size() -> None:
    report = build_scale_candidate_report([CONFIG_DIR / "model_354m.yaml"])
    candidate = report.candidates[0]

    assert candidate.fp32_parameter_gb > candidate.bf16_parameter_gb
    assert candidate.bf16_parameter_gb > candidate.int8_parameter_gb
    assert candidate.rough_training_checkpoint_gb > candidate.fp32_parameter_gb
    assert "lowest candidate burden" in candidate.local_inference_burden


def test_scale_candidate_report_text_keeps_default_recommendation() -> None:
    report = build_scale_candidate_report([CONFIG_DIR / "model_354m.yaml"])
    text = format_report(report)

    assert DEFAULT_RECOMMENDATION in text
    assert "final" not in text.lower()
