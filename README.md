# CodeWriter-Tutor TPU

This repository builds the CodeWriter-Tutor TPU family (`cw360`): specialized Python code writing, debugging, and tutoring models.

Phase 0 is repository foundation only. It includes configuration scaffolding, documentation placeholders, local validation commands, quality tooling, a model-size matrix, and import/config tests. It does not implement model architecture, dataset loading, or training.

## Boundaries

- The model remains specialized for Python code writing, debugging, and tutoring.
- The tokenizer is StarCoder2; no tokenizer is trained from scratch.
- Final training happens on Kaggle TPU from prepacked binary token shards.
- The laptop is for local validation and final CPU inference.
- Full model training does not happen locally.
- Parent LLM data structuring happens offline before training and before final prepacking.
- Real TPU training never tokenizes raw JSONL live.
- Parent/teacher APIs must never be called inside live TPU training.
- Local CPU tests must not require `torch_xla`.

## Local Validation

```bash
python -m pytest tests/test_imports.py tests/test_configs.py -q
python scripts/print_config.py --config configs/model_354m.yaml
python scripts/print_config.py --config configs/model_420m.yaml
python scripts/print_config.py --config configs/model_480m.yaml
make local-test
```

## Model Candidates

The active candidate configs are `354m`, `420m`, and `480m`, plus tiny CPU/TPU configs for local validation. See `docs/MODEL_SCALE_MATRIX.md`.
