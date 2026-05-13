# Teacher Candidates

Candidate mining selects source examples for offline parent LLM structuring. It does not
call teacher APIs inside the live TPU training loop.

## Candidate Rules

- Keep raw text and source metadata for parent-LLM scoring and structuring.
- Prefer Python code writing, debugging, tests, refactor, explanation, and tutoring tasks.
- Preserve candidate IDs so synthetic records can split by candidate and avoid leakage.
- Dry-run and fake-data modes must work locally.

## Commands

```bash
python scripts/preview_teacher_candidates.py --config configs/teacher_candidates.yaml --source fake --limit 10 --use-fake-data
python scripts/mine_teacher_candidates.py --config configs/teacher_candidates.yaml --source fake --limit 20 --max-candidates 20 --output-dir outputs/candidates --dry-run --preview --use-fake-data
```
