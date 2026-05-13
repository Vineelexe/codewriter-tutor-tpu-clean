# Technical Requirements Document

CodeWriter-Tutor TPU is a Python-code-specialized language model family for code writing,
debugging, and tutoring. It is not a general chatbot, agent, RAG system, multimodal system,
or autonomous coding product.

## Product Goal

Train and evaluate a compact decoder-only causal LM that can run locally on CPU while using
Kaggle TPU time for real training. The model family remains between the current 354M, 420M,
and 480M candidates until TPU benchmarking selects the final size.

## Hard Constraints

- Tokenization uses the StarCoder2 tokenizer, currently `bigcode/starcoder2-15b`.
- No tokenizer is trained from scratch.
- Real TPU training consumes prepacked binary token shards only.
- Parent or teacher model calls run offline before instruction tuning or prepacking.
- Unit tests use mock providers and never call real external APIs.
- Local CPU inference is mandatory for the i7 10th Gen, 16 GB RAM target.
- GGUF and custom 4-bit export are optional research items, not promised deliverables.

## TPU Pivot

The project moved away from T4-style streaming training because streaming JSONL, runtime
tokenization, dynamic padding, and CUDA fp16 `GradScaler` assumptions fight the TPU path.
The TPU path needs stable shapes, fast sequential shard reads, deterministic resume cursors,
and bfloat16/XLA behavior.

Final TPU training therefore reads `[N, 1025] uint16` `.npy` shards. Each row stores one
causal LM sequence of `max_seq_len + 1` tokens for a 1024-token context: inputs are
`row[:-1]` and labels are `row[1:]`. The StarCoder2 vocabulary fits in `uint16`, so the
format is compact and simple to memory-map.

## Success Criteria

- CPU tiny tests pass locally.
- TPU dry runs validate static shapes, `drop_last=True`, bf16/XLA readiness, shard manifests,
and checkpoint metadata.
- Every checkpoint has JSON metadata, data lineage, data snapshot hashes, and an exact TPU
  data cursor: epoch, shard order seed, shard index, sequence offset, consumed sequences,
  and tokens seen.
- Vineel and Sarang exchange one legal checkpoint lineage through relay handoff only.
