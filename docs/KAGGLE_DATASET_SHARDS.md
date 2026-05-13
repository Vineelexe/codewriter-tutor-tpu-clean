# Kaggle Dataset Shards

Final prepacked shards are uploaded as a private Kaggle Dataset so Kaggle TPU notebooks read
binary token shards from `/kaggle/input/...` and write checkpoints to `/kaggle/working`.

## Required Contents

- Train shard files.
- Validation shard files.
- Optional test shard files.
- Global `manifest.json`.
- Per-shard metadata.
- Kaggle dataset metadata produced by the prepack writer.

The manifest is part of checkpoint data lineage. Its hashes must be preserved in checkpoint
metadata and relay handoffs.

## Upload Flow

```bash
python scripts/verify_tpu_shards.py --input-dir outputs/tpu_shards_1024
python scripts/upload_kaggle_dataset.py --input-dir outputs/tpu_shards_1024 --create
```

For a new dataset version after a verified re-pack:

```bash
python scripts/upload_kaggle_dataset.py --input-dir outputs/tpu_shards_1024
```

Kaggle quota and runtime values must be checked immediately before launch. Conservative
defaults are acceptable; stale quota claims are not.
