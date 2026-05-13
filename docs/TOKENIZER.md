# Tokenizer

CodeWriter-Tutor uses the StarCoder2 tokenizer family through
`transformers.AutoTokenizer`. The project default is `bigcode/starcoder2-3b`,
while model configs may pin another StarCoder2-compatible tokenizer such as
`bigcode/starcoder2-15b`.

No tokenizer is trained from scratch. Reusing StarCoder2 keeps Python code,
indentation, operators, docstrings, and FIM-style code tokens aligned with the
base coding model family. It also avoids adding a new vocabulary compatibility
risk to relay training and checkpoint recovery.

TPU prepacked shards store token IDs as `uint16` because the StarCoder2
vocabulary fits below `65536`. This halves shard storage compared with `int32`
while preserving exact token IDs. Verification asserts that the tokenizer max
token id is `< 65536`; if that fails, prepacking must not proceed with `uint16`.

Local verification:

```bash
python scripts/verify_tokenizer.py --config configs/model_354m.yaml
python -m pytest tests/test_tokenizer_config.py tests/test_tokenizer_roundtrip.py tests/test_tokenizer_uint16_safety.py -q
```

The Makefile remains for environments with `make`, but Windows validation uses
direct Python commands as the source of truth:

```bash
python -m pytest tests/test_imports.py tests/test_configs.py tests/test_tokenizer_config.py tests/test_tokenizer_roundtrip.py tests/test_tokenizer_uint16_safety.py -q
python scripts/verify_tokenizer.py --config configs/model_354m.yaml
```

If a Hugging Face repository is private, gated, or rate limited, set
`HF_TOKEN` before running verification, or use an existing `huggingface-cli
login` session:

```bash
set HF_TOKEN=hf_your_token_here
python scripts/verify_tokenizer.py --config configs/model_354m.yaml
```

On Kaggle, store the token as a Kaggle secret and export it to `HF_TOKEN` before
running offline parent-data preparation or tokenizer verification. Real TPU
training consumes prepacked binary token shards and must not rely on runtime
padding, raw JSONL streams, or live tokenizer downloads inside the training
loop.
