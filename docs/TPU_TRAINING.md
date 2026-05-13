# TPU Training

Real TPU training reads only prepacked `.npy` token shards. It does not read raw JSONL,
does not call parent LLMs, and does not perform runtime dynamic padding.

## Runtime Decisions

- Use PyTorch/XLA or another explicit TPU backend for TPU paths.
- Use bfloat16/XLA behavior. T4-style CUDA fp16 `GradScaler` logic does not apply.
- Keep `static_shapes: true`, `drop_last: true`, and `dynamic_padding: false`.
- Use uniform causal LM loss over final token shards.
- Apply `loss_weight` only before prepacking through sampling or oversampling.
- Use the deep-thin 1024-token candidate configs: hidden 768, 12 query heads, 4 KV
  heads, head dim 64, MLP 2304, and 46/56/64 layers for 354M/420M/480M.

## Cursor and Resume

Every TPU checkpoint must include the exact data cursor:

- epoch
- shard order seed
- shard index
- sequence offset
- consumed sequences
- tokens seen

This cursor makes resume deterministic after Kaggle session cutoff. Never overwrite a
checkpoint; save a new checkpoint with metadata and verify it before handoff.

## Commands

```bash
python scripts/train_dry_run.py --config configs/train_tpu_354m_1024.yaml --no-tpu-required
python scripts/tpu_config_audit.py --config configs/train_tpu_354m_1024.yaml
python scripts/tpu_train.py --config configs/train_tpu_354m_1024.yaml --prepacked-dir /kaggle/input/cw360-prepacked-1024 --output-dir /kaggle/working/checkpoints --resume latest
```
