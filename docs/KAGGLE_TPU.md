# Kaggle TPU Runbook

Kaggle TPU is the real training target. Treat each session as interruptible: verify setup,
train in background mode, save before exit, upload or copy durable checkpoints, and resume
from the latest verified checkpoint.

## Setup Check

```bash
python scripts/kaggle_tpu_quota_note.py
python scripts/kaggle_tpu_setup_check.py --config configs/train_tpu_354m_1024.yaml --prepacked-dir /kaggle/input/cw360-prepacked-1024 --checkpoint-dir /kaggle/working/checkpoints --dry-run
python scripts/train_dry_run.py --config configs/train_tpu_354m_1024.yaml
```

The setup check must confirm the tokenizer id, prepacked manifest, checkpoint directory,
session time guard, and durable store settings. Verify current Kaggle quota/runtime values
before launch.

## Background Launch

```bash
nohup python scripts/kaggle_tpu_background_entry.py --config configs/train_tpu_354m_1024.yaml --prepacked-dir /kaggle/input/cw360-prepacked-1024 --checkpoint-dir /kaggle/working/checkpoints --durable-dir /kaggle/working/durable --next-trainer sarang > /kaggle/working/train.log 2>&1 &
tail -f /kaggle/working/train.log
```

Background mode keeps training alive while the notebook UI is not the active foreground
process. The entrypoint still writes checkpoints and handoff metadata through the project
scripts.

## Save Before Exit

```bash
python scripts/kaggle_tpu_save_before_exit.py --config configs/train_tpu_354m_1024.yaml --checkpoint-dir /kaggle/working/checkpoints --durable-dir /kaggle/working/durable
python scripts/kaggle_tpu_resume_latest.py --checkpoint-dir /kaggle/working/checkpoints
```

## Resume

```bash
python scripts/kaggle_tpu_resume_latest.py --checkpoint-dir /kaggle/working/checkpoints
python scripts/kaggle_tpu_train.py --config configs/train_tpu_354m_1024.yaml --prepacked-dir /kaggle/input/cw360-prepacked-1024 --checkpoint-dir /kaggle/working/checkpoints --resume latest --durable-dir /kaggle/working/durable --next-trainer sarang
```

Resume only from a checkpoint bundle with metadata JSON, data snapshot metadata, lineage,
and TPU cursor fields.
