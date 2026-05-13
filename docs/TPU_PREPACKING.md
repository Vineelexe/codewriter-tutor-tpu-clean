# TPU Prepacking

TPU prepacking converts validated training examples into binary token shards before real
training starts. It is a CPU/cloud-CPU data factory step, not a live TPU training step.

## Final Shard Format

The real 1024-context TPU format is `[N, 1025] uint16`:

- `N` is the number of fixed-length sequences in the shard.
- `1025` is `max_seq_len + 1`, allowing causal LM inputs and labels from one row.
- `uint16` is valid because the StarCoder2 vocab size in configs is 49,152.
- `.npy` keeps reads simple and avoids parsing overhead in the training loop.

JSONL is not used inside the TPU training loop because it would require live parsing,
tokenization, filtering, padding, and mixture decisions. Those operations are slower,
less recoverable, and harder to resume exactly after a Kaggle session cutoff.

## Static Shape Rules

Real TPU batches must have static shapes. The trainer uses prepacked fixed-length rows and
`drop_last=True`, so the last incomplete batch is discarded instead of changing batch shape.
There is no dynamic padding in real TPU training.

## Tiny Local Prepack

Use a tiny local shard for loader, checkpoint, and resume tests:

```bash
python scripts/make_tiny_prepacked_dataset.py --config configs/prepack_tpu_1024.yaml --output-dir outputs/tiny_prepacked --num-sequences 64
python scripts/verify_tpu_shards.py --input-dir outputs/tiny_prepacked
python scripts/preview_tpu_shards.py --input-dir outputs/tiny_prepacked --split train --limit 3
```

## Real Prepack

Use cloud CPU where possible, with remote structured inputs and enough RAM for rolling
shuffle:

```bash
python scripts/pack_tpu_dataset.py --config configs/prepack_tpu_1024.yaml --datasets-config configs/datasets.yaml --synthetic-input outputs/synthetic/train.jsonl --output-dir outputs/tpu_shards_1024 --target-sequences 1000000 --training-stage base_pretrain
python scripts/verify_tpu_shards.py --input-dir outputs/tpu_shards_1024
python scripts/benchmark_tpu_shard_io.py --input-dir outputs/tpu_shards_1024 --split train --num-batches 20 --batch-size 16
```
