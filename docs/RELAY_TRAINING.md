# Relay Training

Relay training is the legal handoff process for one checkpoint lineage between Vineel and
Sarang. There is no model averaging, checkpoint averaging, model merging, independent model
halves training, or distributed sharding.

## Handoff Rules

- One trainer owns the active checkpoint lineage at a time.
- The trainer saves a new checkpoint before leaving the session.
- The checkpoint bundle includes metadata JSON, data snapshot metadata, lineage, and exact
  TPU data cursor.
- The next trainer resumes from that checkpoint and continues the same lineage.
- Checkpoints are never overwritten.

## Handoff Command

```bash
python scripts/kaggle_tpu_handoff.py --checkpoint /kaggle/working/checkpoints/step_000001000.pt --trainer-leaving vineel --next-trainer sarang --resume-command "python scripts/kaggle_tpu_train.py --config configs/train_tpu_354m_1024.yaml --prepacked-dir /kaggle/input/cw360-prepacked-1024 --checkpoint-dir /kaggle/working/checkpoints --resume latest --durable-dir /kaggle/working/durable --next-trainer vineel"
python scripts/kaggle_tpu_verify_checkpoint.py --checkpoint /kaggle/working/checkpoints/step_000001000.pt
```

The handoff manifest is an operator record; it does not replace checkpoint metadata.
