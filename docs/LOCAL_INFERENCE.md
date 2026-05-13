# Local Inference

The final checkpoint must run on the laptop through PyTorch CPU inference. Do not assume
CUDA, integrated graphics, or GGUF compatibility.

## Export

Export only after a real or tiny test checkpoint exists and has its metadata JSON sidecar:

```bash
python scripts/export_model.py --checkpoint PATH_TO_CHECKPOINT.pt --output-dir exported_model
```

The export writes:

```text
exported_model/
  config.yaml
  model.safetensors
  tokenizer_ref.txt
  generation_config.yaml
  metadata.json
  quantization_report.json
  demo_prompts.json
  run_local.py
```

`metadata.json` preserves checkpoint metadata, `data_snapshot`, and the exact
`tpu_data_cursor` when the checkpoint came from TPU prepacked training.

## Run

```bash
python exported_model/run_local.py --prompt "Write a Python function to add two numbers." --max-new-tokens 64
python exported_model/run_local.py --demo --max-new-tokens 64 --save-output exported_model/demo_outputs.json
```

The runner loads `model.safetensors` on CPU and loads the StarCoder2 tokenizer named in
`tokenizer_ref.txt` / `config.yaml`. If the tokenizer is not already cached, set `HF_TOKEN`
or pre-populate the Hugging Face cache before running.

## Memory And Speed

`quantization_report.json` includes parameter-only memory estimates for FP32, bf16/fp16,
int8, and theoretical 4-bit storage. These are not full process memory guarantees; Python,
PyTorch, activations, tokenizer state, and KV cache add overhead.

The current runtime target is an i7 10th Gen class CPU with 16 GB RAM. A 354M to 480M
model may fit in RAM in FP32, but generation can still be slow. Measure tokens per second
on the actual exported checkpoint before relying on it for daily use.

## Quantization Status

PyTorch dynamic int8 quantization is exposed as an explicit runtime attempt:

```bash
python exported_model/run_local.py --quantization dynamic-int8 --prompt "Explain a Python KeyError."
```

This only targets `torch.nn.Linear` modules and must be checked for quality and speed on
the exported checkpoint. It is not used during TPU training.

bitsandbytes is not enabled for the CPU laptop target. A custom 4-bit loader is not
implemented.

## GGUF Status

GGUF is unsupported in this repo right now. There is no verified converter for the
CodeWriter-Tutor architecture and StarCoder2 tokenizer path, so llama.cpp compatibility is
not promised.

## Limitations

The local runner is for Python code writing, debugging, explanation, tests, comments, and
refactor prompts. It is not a general chatbot runtime. Full TPU training still consumes
prepacked token shards only; parent/teacher generation remains offline and never runs in
the live TPU training loop.
