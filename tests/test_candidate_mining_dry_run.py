from __future__ import annotations

import subprocess
import sys

from scripts.mine_teacher_candidates import mine_candidates


def test_mine_candidates_fake_source_respects_max_candidates() -> None:
    candidates, router = mine_candidates(
        config_path="configs/teacher_candidates.yaml",
        source="fake",
        max_candidates=3,
        use_fake_data=True,
    )

    assert len(candidates) == 3
    assert router.total_selected == 3
    assert {candidate.source for candidate in candidates}.issubset(
        {"stack_v2_python", "codesearchnet_python", "debugbench"}
    )


def test_mining_dry_run_cli_does_not_write_outputs(tmp_path) -> None:
    output_dir = tmp_path / "dry_run_candidates"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/mine_teacher_candidates.py",
            "--config",
            "configs/teacher_candidates.yaml",
            "--source",
            "fake",
            "--max-candidates",
            "2",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "dry_run selected=2" in result.stdout
    assert not output_dir.exists()


def test_preview_cli_prints_required_candidate_fields() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/preview_teacher_candidates.py",
            "--config",
            "configs/teacher_candidates.yaml",
            "--limit",
            "2",
            "--use-fake-data",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "candidate_id=" in result.stdout
    assert "task_type=" in result.stdout
    assert "quality_tier=" in result.stdout
    assert "proposed_train_stage=" in result.stdout
