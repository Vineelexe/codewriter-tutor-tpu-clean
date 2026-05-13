.PHONY: test lint format print-config verify-tokenizer local-test local-smoke local-infer-tiny local-teacher-mock local-train-tiny local-prepack-tiny local-verify-prepack local-count-354m local-count-420m local-count-480m

PYTHON ?= python

test:
	$(PYTHON) -m pytest tests/test_imports.py tests/test_configs.py tests/test_tokenizer_config.py tests/test_tokenizer_roundtrip.py tests/test_tokenizer_uint16_safety.py -q

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

print-config:
	$(PYTHON) scripts/print_config.py --config configs/model_tiny.yaml

verify-tokenizer:
	$(PYTHON) scripts/verify_tokenizer.py --config configs/model_354m.yaml

local-test:
	$(PYTHON) -m pytest tests/test_local_runtime.py tests/test_local_memory.py -q

local-smoke:
	$(PYTHON) scripts/local_smoke.py --config configs/model_tiny.yaml

local-infer-tiny:
	$(PYTHON) scripts/local_infer_tiny.py --config configs/model_tiny.yaml --prompt "Write a tiny Python function."

local-teacher-mock:
	$(PYTHON) scripts/local_runtime_check.py --mode teacher-mock

local-train-tiny:
	$(PYTHON) scripts/local_runtime_check.py --mode train-tiny --config configs/model_tiny.yaml --steps 20

local-prepack-tiny:
	$(PYTHON) scripts/local_runtime_check.py --mode prepack-tiny --config configs/model_tiny.yaml --steps 20

local-verify-prepack:
	@echo "prepack verification is deferred; configs require train/validation shard outputs"

local-count-354m:
	$(PYTHON) scripts/local_memory_check.py --config configs/model_354m.yaml --no-forward

local-count-420m:
	$(PYTHON) scripts/local_memory_check.py --config configs/model_420m.yaml --no-forward

local-count-480m:
	$(PYTHON) scripts/local_memory_check.py --config configs/model_480m.yaml --no-forward
