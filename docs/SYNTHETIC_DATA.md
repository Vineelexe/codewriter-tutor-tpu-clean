# Synthetic Data

Synthetic records are structured offline before prepacking. Parent LLM outputs must be
validated, deduplicated, split, and stored remotely before they can influence real TPU
training.

## Requirements

- Parent LLM calls happen before instruction tuning or prepacking.
- Tests use mock providers, never real APIs.
- Records stay Python-code-focused.
- Structured shards should be stored in a private Hugging Face Dataset repo where possible.
- Bad, duplicate, or schema-invalid records are rejected before prepack.

## Commands

```bash
python scripts/cloud_parent_llm_generate.py --config configs/teacher_generation.yaml --provider mock --max-requests 10 --local-only
python scripts/validate_synthetic_data.py --input outputs/synthetic/train.jsonl --output-report outputs/synthetic/validation_report.json
python scripts/synthetic_data_report.py --input outputs/synthetic/train.jsonl
python scripts/split_synthetic_data.py --input outputs/synthetic/all.jsonl --output-dir outputs/synthetic_split
```
