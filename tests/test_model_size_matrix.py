from __future__ import annotations

from pathlib import Path

import pytest

from cw360.config import load_model_config
from cw360.model.count import build_parameter_report, validate_size_label_count

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("model_354m.yaml", 354_000_000),
        ("model_420m.yaml", 423_000_000),
        ("model_480m.yaml", 479_000_000),
    ],
)
def test_model_size_labels_match_no_forward_parameter_estimates(
    filename: str,
    expected: int,
) -> None:
    config = load_model_config(CONFIG_DIR / filename)

    report = build_parameter_report(config)

    assert report.expected_parameters == expected
    assert validate_size_label_count(report, tolerance=0.01)


def test_deep_thin_parameter_counts_match_current_matrix() -> None:
    counts = {
        filename: build_parameter_report(load_model_config(CONFIG_DIR / filename)).total_parameters
        for filename in ["model_354m.yaml", "model_420m.yaml", "model_480m.yaml"]
    }

    assert counts == {
        "model_354m.yaml": 354_417_920,
        "model_420m.yaml": 423_258_880,
        "model_480m.yaml": 478_331_648,
    }


def test_model_size_matrix_supports_multiple_non_tiny_configs() -> None:
    counts = [
        build_parameter_report(load_model_config(CONFIG_DIR / filename)).total_parameters
        for filename in ["model_354m.yaml", "model_420m.yaml", "model_480m.yaml"]
    ]

    assert counts == sorted(counts)
    assert len(set(counts)) == 3
