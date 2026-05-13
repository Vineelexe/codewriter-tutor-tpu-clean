# Troubleshooting

## Tokenizer Fails

Run:

```bash
python scripts/verify_tokenizer.py --config configs/model_354m.yaml
```

The tokenizer must be StarCoder2. Do not replace it with a newly trained tokenizer.

## TPU Shape Errors

Check that shards are `[N, 1025] uint16`, the train config has `static_shapes: true`,
`drop_last: true`, and `dynamic_padding: false`, and the loader is reading prepacked
binary shards.

```bash
python scripts/verify_tpu_shards.py --input-dir outputs/tpu_shards_1024
python scripts/tpu_config_audit.py --config configs/train_tpu_354m_1024.yaml
```

## Slow or Unstable TPU Input

Verify mixture-balanced shuffled shards and benchmark sequential reads:

```bash
python scripts/benchmark_tpu_shard_io.py --input-dir outputs/tpu_shards_1024 --split train --num-batches 20 --batch-size 16
```

## Resume Does Not Match

Only resume from a checkpoint with complete metadata and an exact TPU data cursor:

```bash
python scripts/kaggle_tpu_resume_latest.py --checkpoint-dir /kaggle/working/checkpoints
python scripts/kaggle_tpu_verify_checkpoint.py --checkpoint /kaggle/working/checkpoints/step_000001000.pt
```

## Local Inference Too Heavy

Use the deployment ladder in `docs/LOCAL_RUNTIME.md`. CPU PyTorch inference is the baseline.
GGUF or custom 4-bit export is not assumed.
