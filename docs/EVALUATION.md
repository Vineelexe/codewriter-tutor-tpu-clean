# Evaluation

Evaluation tracks whether checkpoints improve Python code writing, debugging, testing,
refactoring, explanation, and tutoring behavior. It is not a general-chat benchmark suite.

## Fixed Prompt Sets

Use versioned JSONL prompt files under `eval_prompts/`. Keep prompt IDs stable so results
can be compared across checkpoints and model sizes.

## Commands

```bash
python scripts/eval_checkpoint.py --config configs/model_354m.yaml --checkpoint /kaggle/working/checkpoints/step_000001000.pt --prompt-file eval_prompts/core_prompts.jsonl --output-dir eval_outputs --device cpu
python scripts/run_eval.py --eval-config configs/eval.yaml --model-config configs/model_354m.yaml --checkpoint /kaggle/working/checkpoints/step_000001000.pt --output-dir eval_outputs
python scripts/compare_eval_runs.py eval_outputs/run_a.jsonl eval_outputs/run_b.jsonl --output eval_outputs/comparison.md
```

Compare validation loss, fixed prompt behavior, and local CPU inference behavior before
selecting between 354M, 420M, and 480M.
