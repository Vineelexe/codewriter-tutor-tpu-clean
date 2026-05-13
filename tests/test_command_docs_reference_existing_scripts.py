from __future__ import annotations

import re
from pathlib import Path


REFERENCE_PATTERNS = [
    re.compile(r"\bscripts/[A-Za-z0-9_./-]+\.py\b"),
    re.compile(r"\bconfigs/[A-Za-z0-9_./-]+\.ya?ml\b"),
    re.compile(r"\btests/[A-Za-z0-9_./-]+\.py\b"),
    re.compile(r"\beval_prompts/[A-Za-z0-9_./-]+\.jsonl\b"),
]


def test_commands_doc_references_existing_repo_files() -> None:
    text = Path("docs/COMMANDS.md").read_text(encoding="utf-8")
    references: set[str] = set()
    for pattern in REFERENCE_PATTERNS:
        references.update(pattern.findall(text))

    assert references, "COMMANDS.md should reference concrete repo scripts/configs/tests"

    missing = sorted(ref for ref in references if not Path(ref).is_file())
    assert missing == []


def test_commands_doc_uses_existing_scale_configs() -> None:
    text = Path("docs/COMMANDS.md").read_text(encoding="utf-8")
    for config in (
        "configs/train_tpu_354m_1024.yaml",
        "configs/train_tpu_420m_1024.yaml",
        "configs/train_tpu_480m_1024.yaml",
        "configs/prepack_tpu_1024.yaml",
    ):
        assert config in text
        assert Path(config).is_file()
