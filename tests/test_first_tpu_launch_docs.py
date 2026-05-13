from __future__ import annotations

import re
from pathlib import Path


DOCS = [
    Path("FIRST_TPU_LAUNCH_PLAN.md"),
    Path("FIRST_TPU_LAUNCH_CHECKLIST.md"),
    Path("FIRST_TPU_RELAY_HANDOFF_TEMPLATE.md"),
]

REQUIRED_COMMANDS = [
    "python scripts/local_smoke.py --config configs/model_tiny.yaml --prompt \"Write a Python function that parses a list of integers.\" --steps 20",
    "python scripts/verify_tokenizer.py --config configs/model_354m.yaml --require-pad-for-batching",
    "python scripts/make_tiny_prepacked_dataset.py --config configs/prepack_tpu_1024.yaml --output-dir outputs/tiny_tpu_shards --num-sequences 64",
    "python scripts/train_tiny_prepacked.py --config configs/train_tiny_prepacked.yaml --prepacked-dir outputs/tiny_tpu_shards --output-dir outputs/tiny_prepacked_train/phase16-smoke",
    "python scripts/verify_tpu_shards.py --input-dir outputs/tiny_tpu_shards",
    "python scripts/verify_tpu_shards.py --input-dir /kaggle/input/cw360-prepacked-1024",
    "python scripts/kaggle_tpu_setup_check.py --dry-run --config configs/train_tpu_354m_1024.yaml",
    "python scripts/train_dry_run.py --config configs/train_tpu_354m_1024.yaml --no-tpu-required",
    "python scripts/kaggle_tpu_background_entry.py --config configs/train_tpu_354m_1024.yaml --prepacked-dir /kaggle/input/cw360-prepacked-1024 --checkpoint-dir /kaggle/working/checkpoints --next-trainer sarang",
    "python scripts/kaggle_tpu_save_before_exit.py --config configs/train_tpu_354m_1024.yaml --checkpoint-dir /kaggle/working/checkpoints",
    "python scripts/kaggle_tpu_resume_latest.py --checkpoint-dir /kaggle/working/checkpoints",
    "python scripts/kaggle_tpu_train.py --config configs/train_tpu_354m_1024.yaml --prepacked-dir /kaggle/input/cw360-prepacked-1024 --checkpoint-dir /kaggle/working/checkpoints --resume latest --next-trainer sarang",
    "python scripts/eval_checkpoint.py --config configs/model_354m.yaml --checkpoint /kaggle/working/checkpoints/CHECKPOINT.pt --prompt-file eval_prompts/core_prompts.jsonl --output-dir outputs/eval/first_tpu_checkpoint --device cpu",
    "python -m pytest tests/ -q",
]

REQUIRED_STOP_CONDITIONS = [
    "NaN",
    "Repeated OOM",
    "Checkpoint save failure",
    "Wall-clock checkpoint failure",
    "Durable upload",
    "Checkpoint transfer failure",
    "Unsupported precision",
    "Dataset access failure",
    "Tokenizer failure",
    "Synthetic data validation failure",
    "Parent-LLM generation storing invalid records",
    "Remote shard upload failure",
    "Missing `HF_TOKEN`",
    "Shard manifest mismatch",
    "Prepacked shard hash mismatch",
    "Dynamic shape detected",
]


def test_first_tpu_launch_docs_exist_and_are_substantive() -> None:
    for path in DOCS:
        assert path.is_file(), f"missing {path}"
        assert len(path.read_text(encoding="utf-8").strip()) > 500


def test_first_tpu_launch_plan_lists_exact_commands() -> None:
    text = Path("FIRST_TPU_LAUNCH_PLAN.md").read_text(encoding="utf-8")
    for command in REQUIRED_COMMANDS:
        assert command in text


def test_first_tpu_launch_docs_cover_readiness_gates_and_stop_conditions() -> None:
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    required_phrases = [
        "Do not train in this phase",
        "StarCoder2",
        "uint16",
        "max_seq_len + 1",
        "kaggle-tpu-background",
        "bf16",
        "drop_last",
        "dynamic_padding",
        "data snapshot",
        "TPU cursor",
        "epoch",
        "shard_order_seed",
        "shard_index",
        "sequence_offset",
        "consumed_sequences",
        "tokens_seen",
        "Kaggle TPU quota",
        "Save Version / Run All",
        "one legal relay lineage",
    ]
    for phrase in required_phrases:
        assert phrase in corpus
    for condition in REQUIRED_STOP_CONDITIONS:
        assert condition in corpus


def test_first_tpu_launch_docs_reference_existing_repo_files() -> None:
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    references = set(
        re.findall(
            r"\b(?:scripts|configs|eval_prompts)/[A-Za-z0-9_./-]+\.(?:py|ya?ml|jsonl)\b",
            corpus,
        )
    )
    assert references
    missing = sorted(ref for ref in references if not Path(ref).is_file())
    assert missing == []


def test_verify_tpu_shards_script_bootstraps_repo_imports() -> None:
    text = Path("scripts/verify_tpu_shards.py").read_text(encoding="utf-8")
    assert "ROOT = Path(__file__).resolve().parents[1]" in text
    assert "sys.path.insert(0, str(ROOT))" in text
