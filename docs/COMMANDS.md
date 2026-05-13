# Commands

These are operator command templates. Replace output directories, checkpoint names, Kaggle
dataset slugs, and private repo IDs with the current run values. Do not paste secrets into
commands.

## 1. Local Setup

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests/test_imports.py tests/test_configs.py -q
```

## 2. Tokenizer Check

```bash
python scripts/verify_tokenizer.py --config configs/model_354m.yaml
python -m pytest tests/test_tokenizer_config.py tests/test_tokenizer_roundtrip.py tests/test_tokenizer_uint16_safety.py -q
```

## 3. Tiny Model Test

```bash
python scripts/smoke_forward.py --config configs/model_tiny.yaml
python scripts/train_tiny_smoke.py --config configs/train_tiny.yaml --output-dir outputs/tiny_train
python scripts/local_infer_tiny.py --config configs/model_tiny.yaml --prompt "Write a tiny Python function."
```

## 4. Candidate Mining Dry Run

```bash
python scripts/preview_teacher_candidates.py --config configs/teacher_candidates.yaml --source fake --limit 10 --use-fake-data
python scripts/mine_teacher_candidates.py --config configs/teacher_candidates.yaml --source fake --limit 20 --max-candidates 20 --output-dir outputs/candidates --dry-run --preview --use-fake-data
```

## 5. Parent LLM Estimate

```bash
python scripts/cloud_parent_llm_estimate.py --config configs/teacher_generation.yaml
python scripts/cloud_parent_llm_status.py --config configs/teacher_generation.yaml
```

## 6. Mock Teacher Generation

```bash
python scripts/cloud_parent_llm_generate.py --config configs/teacher_generation.yaml --provider mock --max-requests 10 --local-only
python scripts/cloud_parent_llm_push_shards.py --config configs/teacher_generation.yaml --mock
```

## 7. Synthetic Validation

```bash
python scripts/validate_synthetic_data.py --input outputs/synthetic/train.jsonl --output-report outputs/synthetic/validation_report.json
python scripts/synthetic_data_report.py --input outputs/synthetic/train.jsonl
python scripts/split_synthetic_data.py --input outputs/synthetic/all.jsonl --output-dir outputs/synthetic_split
```

## 8. Tiny Prepack

```bash
python scripts/make_tiny_prepacked_dataset.py --config configs/prepack_tpu_1024.yaml --output-dir outputs/tiny_prepacked --num-sequences 64
python scripts/verify_tpu_shards.py --input-dir outputs/tiny_prepacked
python scripts/preview_tpu_shards.py --input-dir outputs/tiny_prepacked --split train --limit 3
```

## 9. Real Prepack

```bash
python scripts/pack_tpu_dataset.py --config configs/prepack_tpu_1024.yaml --datasets-config configs/datasets.yaml --synthetic-input outputs/synthetic/train.jsonl --output-dir outputs/tpu_shards_1024 --target-sequences 1000000 --training-stage base_pretrain
```

## 10. Shard Verification

```bash
python scripts/verify_tpu_shards.py --input-dir outputs/tpu_shards_1024
python scripts/benchmark_tpu_shard_io.py --input-dir outputs/tpu_shards_1024 --split train --num-batches 20 --batch-size 16
```

## 11. Kaggle Dataset Upload

```bash
python scripts/upload_kaggle_dataset.py --input-dir outputs/tpu_shards_1024 --create
python scripts/upload_kaggle_dataset.py --input-dir outputs/tpu_shards_1024
```

## 12. Kaggle TPU Setup Check

```bash
python scripts/kaggle_tpu_quota_note.py
python scripts/kaggle_tpu_setup_check.py --config configs/train_tpu_354m_1024.yaml --prepacked-dir /kaggle/input/cw360-prepacked-1024 --checkpoint-dir /kaggle/working/checkpoints --dry-run
python scripts/tpu_config_audit.py --config configs/train_tpu_354m_1024.yaml
```

## 13. TPU Dry Run

```bash
python scripts/train_dry_run.py --config configs/train_tpu_354m_1024.yaml --no-tpu-required
python scripts/train_dry_run.py --config configs/train_tpu_420m_1024.yaml --no-tpu-required
python scripts/train_dry_run.py --config configs/train_tpu_480m_1024.yaml --no-tpu-required
```

## 14. Background Training Launch

```bash
nohup python scripts/kaggle_tpu_background_entry.py --config configs/train_tpu_354m_1024.yaml --prepacked-dir /kaggle/input/cw360-prepacked-1024 --checkpoint-dir /kaggle/working/checkpoints --durable-dir /kaggle/working/durable --next-trainer sarang > /kaggle/working/train.log 2>&1 &
tail -f /kaggle/working/train.log
```

## 15. Save Before Exit

```bash
python scripts/kaggle_tpu_save_before_exit.py --config configs/train_tpu_354m_1024.yaml --checkpoint-dir /kaggle/working/checkpoints --durable-dir /kaggle/working/durable
python scripts/kaggle_tpu_verify_checkpoint.py --checkpoint /kaggle/working/checkpoints/step_000001000.pt
```

## 16. Resume Latest Checkpoint

```bash
python scripts/kaggle_tpu_resume_latest.py --checkpoint-dir /kaggle/working/checkpoints
python scripts/kaggle_tpu_train.py --config configs/train_tpu_354m_1024.yaml --prepacked-dir /kaggle/input/cw360-prepacked-1024 --checkpoint-dir /kaggle/working/checkpoints --resume latest --durable-dir /kaggle/working/durable --next-trainer sarang
```

## 17. Relay Handoff

```bash
python scripts/kaggle_tpu_handoff.py --checkpoint /kaggle/working/checkpoints/step_000001000.pt --trainer-leaving vineel --next-trainer sarang --resume-command "python scripts/kaggle_tpu_train.py --config configs/train_tpu_354m_1024.yaml --prepacked-dir /kaggle/input/cw360-prepacked-1024 --checkpoint-dir /kaggle/working/checkpoints --resume latest --durable-dir /kaggle/working/durable --next-trainer vineel"
```

## 18. Eval Checkpoint

```bash
python scripts/eval_checkpoint.py --config configs/model_354m.yaml --checkpoint /kaggle/working/checkpoints/step_000001000.pt --prompt-file eval_prompts/core_prompts.jsonl --output-dir eval_outputs --device cpu
python scripts/compare_eval_runs.py eval_outputs/run_a.jsonl eval_outputs/run_b.jsonl --output eval_outputs/comparison.md
```

## 19. Local Inference Export

The reliable local path is PyTorch CPU inference from a safetensors export. Dynamic int8 is
an optional runtime attempt. GGUF is unsupported unless a real converter is added and
verified.

```bash
python scripts/export_model.py --checkpoint PATH_TO_CHECKPOINT.pt --output-dir exported_model
python exported_model/run_local.py --prompt "Write a Python function that validates an email address." --max-new-tokens 128
python exported_model/run_local.py --demo --max-new-tokens 64 --save-output exported_model/demo_outputs.json
python scripts/local_memory_check.py --config configs/model_354m.yaml --no-forward
```
