# Architecture

The system is split into offline data preparation, prepacking, TPU training, checkpoint
relay, evaluation, and local CPU inference.

```text
raw/code datasets
  -> candidate mining
  -> offline parent LLM structuring
  -> synthetic validation and splitting
  -> CPU prepacking into train/val/test .npy shards
  -> private Kaggle Dataset
  -> Kaggle TPU training from prepacked shards
  -> checkpoint + metadata + data cursor
  -> relay handoff between Vineel and Sarang
  -> eval and local CPU inference
```

## Boundaries

- `cw360/data` handles local extraction, formatting, mixture sampling, and stream modes.
- `cw360/teacher` handles offline parent/teacher generation with mockable providers.
- `cw360/prepack` writes and verifies final TPU token shards and manifests.
- `cw360/train` contains CPU and TPU training logic, including static-shape validation.
- `cw360/checkpoint` owns no-overwrite checkpoint naming, metadata, lineage, and cursors.
- `cw360/kaggle` owns Kaggle relay support and durable checkpoint-store checks.
- `cw360/inference` is the local CPU inference path.

## Model Family

The active TPU launch candidates are the original deep-thin family:

| Candidate | Hidden | Layers | Q Heads | KV Heads | Head Dim | MLP | Seq Len |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 354M | 768 | 46 | 12 | 4 | 64 | 2304 | 1024 |
| 420M | 768 | 56 | 12 | 4 | 64 | 2304 | 1024 |
| 480M | 768 | 64 | 12 | 4 | 64 | 2304 | 1024 |

The matrix keeps StarCoder2 tokenization, RMSNorm, RoPE, GQA, SwiGLU, QKV bias, and
tied embeddings fixed while TPU benchmarking decides between the sizes.

## Non-Goals

The architecture excludes agents, RAG, live tool use, web browsing, model merging,
checkpoint averaging, distributed model sharding, MoE, multimodal features, and autonomous
PR or coding behavior.
