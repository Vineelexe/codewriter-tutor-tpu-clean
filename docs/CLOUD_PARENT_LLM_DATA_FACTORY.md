# Cloud Parent LLM Data Factory

The parent LLM data factory runs offline, preferably on cloud CPU environments. Its output
is structured synthetic data for later validation and prepacking. It is not part of the TPU
training loop.

## Flow

1. Estimate request volume and cost.
2. Generate with a real or mock provider outside TPU training.
3. Write resumable manifests and local temp shards.
4. Push validated structured shards to private remote storage.
5. Resume from manifests after interruption.

## Commands

```bash
python scripts/cloud_parent_llm_estimate.py --config configs/teacher_generation.yaml
python scripts/cloud_parent_llm_generate.py --config configs/teacher_generation.yaml --provider mock --max-requests 10 --local-only
python scripts/cloud_parent_llm_status.py --config configs/teacher_generation.yaml
python scripts/cloud_parent_llm_resume.py --config configs/teacher_generation.yaml --provider mock --max-requests 10 --local-only
python scripts/cloud_parent_llm_push_shards.py --config configs/teacher_generation.yaml --mock
```

Real providers must have mock equivalents for tests. Do not hardcode secrets; use the
environment handling defined by the project.
