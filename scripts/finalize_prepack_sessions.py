from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cw360.prepack.config import PrepackFactoryConfig, load_prepack_config  # noqa: E402
from cw360.prepack.kaggle_dataset import kaggle_slug, write_kaggle_dataset_metadata  # noqa: E402
from cw360.prepack.manifest import (  # noqa: E402
    current_git_commit,
    hash_payload,
    merge_count_dicts,
    normalize_counts,
    read_json,
    sha256_file,
    utc_now_iso,
    write_json,
)
from cw360.prepack.session_runner import (  # noqa: E402
    ValidPrepackSession,
    scan_prepack_sessions,
)
from cw360.prepack.shard_index import SPLITS  # noqa: E402
from cw360.prepack.verify import verify_prepacked_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finalize multiple safe prepack sessions into one trainer-ready dataset."
    )
    parser.add_argument("--sessions-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--require-target-total-sequences", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_prepack_config(args.config) if args.config else None
    scan = scan_prepack_sessions(args.sessions_root)
    completed_invalid = [session for session in scan.invalid_sessions if session.completed]
    if completed_invalid:
        details = "; ".join(
            f"{session.session_id}: {session.reason}" for session in completed_invalid
        )
        raise RuntimeError(f"completed prepack session validation failed: {details}")
    if not scan.valid_sessions:
        raise RuntimeError("no valid prepack sessions found")
    if (
        args.require_target_total_sequences is not None
        and scan.total_valid_sequences < args.require_target_total_sequences
    ):
        raise RuntimeError(
            "valid prepack sessions are below required target: "
            f"{scan.total_valid_sequences} < {args.require_target_total_sequences}"
        )

    output_dir = Path(args.output_dir)
    _prepare_output_dir(output_dir, overwrite=args.overwrite)
    manifest = finalize_sessions(
        scan.valid_sessions,
        output_dir=output_dir,
        config=config,
    )
    _write_dataset_metadata(output_dir, manifest, config)
    report = verify_prepacked_dataset(output_dir)
    print(
        "finalized "
        f"{len(scan.valid_sessions)} sessions into {output_dir} with "
        f"{report.shard_count} shards and {report.total_sequences} sequences"
    )
    return 0


def finalize_sessions(
    sessions: list[ValidPrepackSession],
    *,
    output_dir: str | Path,
    config: PrepackFactoryConfig | None = None,
) -> dict[str, Any]:
    if not sessions:
        raise ValueError("at least one valid session is required")
    output_root = Path(output_dir)
    _ensure_compatible_sessions(sessions)
    if config is not None:
        _ensure_config_compatible(config, sessions[0].manifest)
    for split in SPLITS:
        (output_root / split).mkdir(parents=True, exist_ok=True)

    base = sessions[0].manifest
    shard_file_list: dict[str, list[str]] = {split: [] for split in SPLITS}
    shard_hashes: dict[str, dict[str, str]] = {split: {} for split in SPLITS}
    shard_rows: list[dict[str, Any]] = []
    counters = {split: 0 for split in SPLITS}
    source_dataset_versions = _merge_source_dataset_versions(sessions)
    session_provenance = _session_provenance(sessions)

    for session in sessions:
        for row in session.manifest.get("shards", []):
            if not isinstance(row, dict):
                raise RuntimeError(f"{session.session_id} contains a non-object shard row")
            split = str(row["split"])
            if split not in SPLITS:
                raise RuntimeError(f"{session.session_id} contains unsupported split: {split}")
            source_rel = str(row["shard_filename"])
            source_npy = session.path / source_rel
            dest_rel = f"{split}/chunk_{counters[split]:06d}.npy"
            counters[split] += 1
            dest_npy = output_root / dest_rel
            _copy_or_hardlink(source_npy, dest_npy)
            digest = sha256_file(dest_npy)
            expected_digest = str(row.get("sha256") or "")
            if digest != expected_digest:
                raise RuntimeError(f"shard hash mismatch after copy: {source_npy}")

            source_metadata = read_json(source_npy.with_suffix(".json"))
            metadata = dict(source_metadata)
            metadata.update(
                {
                    "shard_filename": dest_rel,
                    "split": split,
                    "sha256": digest,
                    "source_session_id": session.session_id,
                    "source_session_shard_filename": source_rel,
                }
            )
            write_json(dest_npy.with_suffix(".json"), metadata)
            shard_rows.append(metadata)
            shard_file_list[split].append(dest_rel)
            shard_hashes[split][dest_rel] = digest

    split_counts = {
        split: sum(int(row["num_sequences"]) for row in shard_rows if row["split"] == split)
        for split in SPLITS
    }
    total_sequences = sum(split_counts.values())
    source_counts = merge_count_dicts([row.get("source_counts", {}) for row in shard_rows])
    per_shard_summary = {
        row["shard_filename"]: row.get(
            "source_mixture_actual",
            normalize_counts(row.get("source_counts", {})),
        )
        for row in shard_rows
    }
    prepack = None if config is None else config.prepack
    validation = _validation_from_config(config) if config is not None else base["validation"]
    manifest: dict[str, Any] = {
        "dataset_name": prepack.dataset_name if prepack is not None else base["dataset_name"],
        "dataset_version": prepack.dataset_version
        if prepack is not None
        else base["dataset_version"],
        "created_at": utc_now_iso(),
        "tokenizer_id": prepack.tokenizer_id if prepack is not None else base["tokenizer_id"],
        "tokenizer_config_hash": base.get("tokenizer_config_hash"),
        "max_seq_len": prepack.max_seq_len if prepack is not None else base["max_seq_len"],
        "seq_len_plus_one": prepack.seq_len_plus_one
        if prepack is not None
        else base["seq_len_plus_one"],
        "dtype": prepack.token_dtype if prepack is not None else base["dtype"],
        "shard_format": prepack.shard_format if prepack is not None else base["shard_format"],
        "shard_count": len(shard_rows),
        "total_sequences": total_sequences,
        "total_tokens_for_training": total_sequences * int(base["max_seq_len"]),
        "total_token_ids_stored": total_sequences * int(base["seq_len_plus_one"]),
        "shard_file_list": shard_file_list,
        "shard_hashes": shard_hashes,
        "split_fractions": _split_fractions(config, base),
        "actual_split_counts": split_counts,
        "actual_split_fractions": normalize_counts(split_counts),
        "target_mixture": base["target_mixture"],
        "actual_global_mixture": normalize_counts(source_counts),
        "per_shard_mixture_summary": per_shard_summary,
        "validation": validation,
        "loss_weight_handling_policy": base["loss_weight_handling_policy"],
        "effective_sampling_multipliers": base.get("effective_sampling_multipliers", {}),
        "source_dataset_versions": source_dataset_versions,
        "synthetic_repo_id": base.get("synthetic_repo_id"),
        "synthetic_manifest_id": base.get("synthetic_manifest_id"),
        "split_manifest_id": base.get("split_manifest_id"),
        "prepack_config_hash": (
            hash_payload(config.to_dict()) if config is not None else base["prepack_config_hash"]
        ),
        "data_snapshot_hash": hash_payload(
            {
                "session_manifest_hashes": [
                    session.manifest.get("manifest_hash") for session in sessions
                ],
                "session_ids": [session.session_id for session in sessions],
            }
        ),
        "code_git_commit": current_git_commit(ROOT),
        "session_provenance": session_provenance,
        "shards": shard_rows,
    }
    manifest["manifest_hash"] = hash_payload(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    write_json(output_root / "manifest.json", manifest)
    return manifest


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is non-empty; pass --overwrite to replace it: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _ensure_compatible_sessions(sessions: list[ValidPrepackSession]) -> None:
    reference = sessions[0].manifest
    keys = (
        "tokenizer_id",
        "max_seq_len",
        "seq_len_plus_one",
        "dtype",
        "shard_format",
        "loss_weight_handling_policy",
    )
    for session in sessions[1:]:
        for key in keys:
            if session.manifest.get(key) != reference.get(key):
                raise RuntimeError(
                    f"{session.session_id} has incompatible {key}: "
                    f"{session.manifest.get(key)!r} != {reference.get(key)!r}"
                )


def _ensure_config_compatible(
    config: PrepackFactoryConfig,
    manifest: dict[str, Any],
) -> None:
    expected = {
        "tokenizer_id": config.prepack.tokenizer_id,
        "max_seq_len": config.prepack.max_seq_len,
        "seq_len_plus_one": config.prepack.seq_len_plus_one,
        "dtype": config.prepack.token_dtype,
        "shard_format": config.prepack.shard_format,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"finalize config is incompatible with sessions for {key}: "
                f"{value!r} != {manifest.get(key)!r}"
            )


def _merge_source_dataset_versions(sessions: list[ValidPrepackSession]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for session in sessions:
        versions = session.manifest.get("source_dataset_versions", {})
        if not isinstance(versions, dict):
            continue
        for key, value in versions.items():
            name = str(key)
            text_value = str(value)
            if name not in merged or merged[name] == text_value:
                merged[name] = text_value
            else:
                merged[f"{session.session_id}:{name}"] = text_value
    return merged


def _session_provenance(sessions: list[ValidPrepackSession]) -> list[dict[str, Any]]:
    return [
        {
            "session_id": session.session_id,
            "session_dir": str(session.path),
            "manifest_hash": session.manifest.get("manifest_hash"),
            "total_sequences": session.report.total_sequences,
            "shard_count": session.report.shard_count,
            "session_info": session.session_info,
        }
        for session in sessions
    ]


def _copy_or_hardlink(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def _validation_from_config(config: PrepackFactoryConfig) -> dict[str, Any]:
    return {
        "shard_mixture_tolerance_abs": config.validation.shard_mixture_tolerance_abs,
        "fail_on_token_overflow": config.validation.fail_on_token_overflow,
        "fail_on_empty_shard": config.validation.fail_on_empty_shard,
        "fail_on_missing_manifest": config.validation.fail_on_missing_manifest,
        "fail_if_split_fractions_do_not_sum_to_one": (
            config.validation.fail_if_split_fractions_do_not_sum_to_one
        ),
        "fail_if_atomic_examples_cross_sequence_boundary": (
            config.validation.fail_if_atomic_examples_cross_sequence_boundary
        ),
    }


def _split_fractions(
    config: PrepackFactoryConfig | None,
    base_manifest: dict[str, Any],
) -> dict[str, float]:
    if config is None:
        return dict(base_manifest["split_fractions"])
    return {
        "train": config.splits.train_fraction,
        "val": config.splits.val_fraction,
        "test": config.splits.test_fraction,
    }


def _write_dataset_metadata(
    output_dir: Path,
    manifest: dict[str, Any],
    config: PrepackFactoryConfig | None,
) -> None:
    if config is not None:
        write_kaggle_dataset_metadata(output_dir, config)
        return
    dataset_name = str(manifest["dataset_name"])
    write_json(
        output_dir / "dataset-metadata.json",
        {
            "title": dataset_name,
            "id": f"vineel/{kaggle_slug(dataset_name)}",
            "licenses": [{"name": "other"}],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
