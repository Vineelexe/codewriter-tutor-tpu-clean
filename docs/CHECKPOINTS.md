# Checkpoints

Checkpoints are append-only training records. Never overwrite a checkpoint path.

## Required Bundle

Every checkpoint must include:

- model weights
- metadata JSON
- checkpoint lineage
- trainer identity
- training stage
- model size label
- data lineage and data snapshot metadata
- prepacked manifest or shard manifest hash
- TPU data cursor with epoch, shard order seed, shard index, sequence offset, consumed
  sequences, and tokens seen

## Verification

```bash
python scripts/kaggle_tpu_verify_checkpoint.py --checkpoint /kaggle/working/checkpoints/step_000001000.pt
python scripts/kaggle_tpu_resume_latest.py --checkpoint-dir /kaggle/working/checkpoints
```

If metadata is missing or the cursor is incomplete, the checkpoint is not valid for relay
training.
