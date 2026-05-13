from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from cw360.data.dedupe import ExactDeduper
from cw360.data.filters import estimate_token_count
from cw360.data.quality_tiers import apply_decision
from cw360.data.source_filters import filter_training_example
from cw360.data.sources import DatasetSourceConfig, load_dataset_source_configs
from cw360.data.streaming import iter_source_records


def _resolve_local_path(config: DatasetSourceConfig, config_path: Path) -> DatasetSourceConfig:
    if not config.local_path:
        return config
    local_path = Path(config.local_path)
    if not local_path.is_absolute():
        local_path = config_path.resolve().parent.parent / local_path
    return DatasetSourceConfig(
        name=config.name,
        dataset_id=config.dataset_id,
        subset=config.subset,
        split=config.split,
        streaming=config.streaming,
        extractor=config.extractor,
        local_path=str(local_path),
        gated=config.gated,
        max_preview_examples=config.max_preview_examples,
        extra=config.extra,
    )


def preview_dataset(config_path: str | Path, source: str, limit: int) -> list[str]:
    config_file = Path(config_path)
    configs = load_dataset_source_configs(config_file)
    if source not in configs:
        raise ValueError(
            f"unknown source {source}; available sources: {', '.join(sorted(configs))}"
        )
    config = _resolve_local_path(configs[source], config_file)
    deduper = ExactDeduper()
    rows: list[str] = []
    for index, record in enumerate(islice(iter_source_records(config), limit), start=1):
        if record.error:
            rows.append(f"{index}. source={source} status=rejected tier=C reason={record.error}")
            continue
        if record.example is None:
            rows.append(f"{index}. source={source} status=rejected tier=C reason=missing_example")
            continue
        decision = filter_training_example(record.example, deduper=deduper)
        apply_decision(record.example, decision)
        snippet = record.example.text.replace("\n", "\\n")[:120]
        rows.append(
            f"{index}. source={record.example.source} status={decision.status} "
            f"tier={decision.quality_tier.value} reason={decision.reason} "
            f"estimated_tokens={estimate_token_count(record.example.text)} text={snippet}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Phase 6 streaming dataset examples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    for row in preview_dataset(args.config, args.source, args.limit):
        print(row)


if __name__ == "__main__":
    main()
