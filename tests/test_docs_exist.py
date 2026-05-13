from __future__ import annotations

from pathlib import Path


DOCS = [
    "TRD.md",
    "ARCHITECTURE.md",
    "MODEL_SCALE_MATRIX.md",
    "DATASETS.md",
    "TPU_PREPACKING.md",
    "KAGGLE_DATASET_SHARDS.md",
    "TPU_TRAINING.md",
    "KAGGLE_TPU.md",
    "RELAY_TRAINING.md",
    "CHECKPOINTS.md",
    "EVALUATION.md",
    "SYNTHETIC_DATA.md",
    "TEACHER_CANDIDATES.md",
    "CLOUD_PARENT_LLM_DATA_FACTORY.md",
    "LOCAL_RUNTIME.md",
    "COMMANDS.md",
    "TROUBLESHOOTING.md",
]


def test_phase_14_docs_exist_and_are_not_placeholders() -> None:
    for doc in DOCS:
        path = Path("docs") / doc
        assert path.is_file(), f"missing {path}"
        text = path.read_text(encoding="utf-8")
        assert len(text.strip()) > 120, f"{path} looks like a placeholder"


def test_phase_14_docs_cover_required_operator_topics() -> None:
    corpus = "\n".join((Path("docs") / doc).read_text(encoding="utf-8") for doc in DOCS)
    normalized = corpus.lower()

    required_phrases = [
        "t4-style streaming training",
        "[n, 1025] uint16",
        "jsonl is not used inside the tpu training loop",
        "shuffled and mixture-balanced",
        "drop_last=true",
        "bfloat16/xla",
        "parent llm calls happen before",
        "kaggle dataset",
        "background mode",
        "relay handoff",
        "local cpu inference",
        "gguf",
    ]

    for phrase in required_phrases:
        assert phrase in normalized, f"missing required documentation topic: {phrase}"


def test_commands_doc_has_all_required_runbook_sections() -> None:
    text = Path("docs/COMMANDS.md").read_text(encoding="utf-8").lower()
    sections = [
        "local setup",
        "tokenizer check",
        "tiny model test",
        "candidate mining dry run",
        "parent llm estimate",
        "mock teacher generation",
        "synthetic validation",
        "tiny prepack",
        "real prepack",
        "shard verification",
        "kaggle dataset upload",
        "kaggle tpu setup check",
        "tpu dry run",
        "background training launch",
        "save before exit",
        "resume latest checkpoint",
        "relay handoff",
        "eval checkpoint",
        "local inference export",
    ]

    for section in sections:
        assert section in text, f"missing COMMANDS.md section: {section}"
