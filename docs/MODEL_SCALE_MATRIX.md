# Model Scale Matrix

The final model size is intentionally unresolved until TPU benchmarks compare throughput,
memory headroom, validation loss, and recovery behavior.

| Candidate | Config | Shape Summary | Parameters | TPU Role |
| --- | --- | --- | --- | --- |
| 354M | `configs/model_354m.yaml` | hidden 768, 46 layers, 12 Q heads, 4 KV heads, head dim 64, MLP 2304, seq 1024 | 354,417,920 | Safest default for first full TPU run. |
| 420M | `configs/model_420m.yaml` | hidden 768, 56 layers, 12 Q heads, 4 KV heads, head dim 64, MLP 2304, seq 1024 | 423,258,880 | Middle candidate if 354M has too much headroom. |
| 480M | `configs/model_480m.yaml` | hidden 768, 64 layers, 12 Q heads, 4 KV heads, head dim 64, MLP 2304, seq 1024 | 478,331,648 | Upper candidate if TPU memory and step time are stable. |

All candidates use the StarCoder2 tokenizer, RMSNorm, RoPE, GQA, SwiGLU, QKV bias,
tied embeddings, and the deep-thin architecture family. The 480M label is acceptable
for the 478.3M-parameter upper candidate because it remains within the intended
candidate range.

## Benchmark Decision Rule

Pick 354M if resume reliability, step time, or memory headroom is tight. Pick 420M if it
fits comfortably and improves validation loss or code eval enough to justify the extra
compute. Pick 480M only if TPU dry runs and background sessions show stable memory,
acceptable tokens/sec, clean checkpoint saves, and better validation/eval results than
420M.

Useful comparison commands:

```bash
python scripts/scale_candidate_report.py --configs configs/model_354m.yaml configs/model_420m.yaml configs/model_480m.yaml
python scripts/train_dry_run.py --config configs/train_tpu_354m_1024.yaml --no-tpu-required
python scripts/train_dry_run.py --config configs/train_tpu_420m_1024.yaml --no-tpu-required
python scripts/train_dry_run.py --config configs/train_tpu_480m_1024.yaml --no-tpu-required
```
