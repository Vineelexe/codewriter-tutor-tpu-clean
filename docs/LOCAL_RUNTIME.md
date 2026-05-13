# Local Runtime

Local inference is mandatory. The runtime target is CPU inference on an i7 10th Gen class
machine with 16 GB RAM; integrated graphics must not be assumed.

## Deployment Ladder

1. PyTorch CPU inference.
2. Safetensors export when supported by the checkpoint/export path.
3. Optional int8 or dynamic quantization after correctness checks.
4. Optional custom 4-bit or GGUF only if honestly implemented and verified.

GGUF is optional and not promised. The current export policy is recorded in
`configs/export.yaml`.

## Commands

```bash
python scripts/local_runtime_check.py --mode smoke --config configs/model_tiny.yaml --steps 20
python scripts/local_memory_check.py --config configs/model_354m.yaml --no-forward
python scripts/local_infer_tiny.py --config configs/model_tiny.yaml --prompt "Write a tiny Python function."
python scripts/generate.py --config configs/model_tiny.yaml --random-model --prompt "Write a tiny Python function that adds two numbers." --device cpu --max-new-tokens 32
```
