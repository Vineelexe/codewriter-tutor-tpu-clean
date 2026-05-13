from __future__ import annotations

import json
import platform
import shutil
import textwrap
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_model

from cw360.checkpoint.manager import torch_load
from cw360.checkpoint.metadata import metadata_path_for_checkpoint, read_metadata_json
from cw360.config import ModelConfig
from cw360.local.memory import build_local_memory_report
from cw360.model.lm import CodeWriterTutorLM
from cw360.tokenizer.special_tokens import is_starcoder2_tokenizer_id

DEFAULT_GENERATION_CONFIG: dict[str, Any] = {
    "max_new_tokens": 128,
    "temperature": 0.0,
    "top_k": None,
    "top_p": None,
    "seed": 1234,
    "use_cache": True,
    "device": "cpu",
}


DEMO_PROMPTS: dict[str, str] = {
    "code_writing": "Write a Python function to add two numbers.",
    "debugging": "Debug this Python function:\n\ndef divide(a, b):\n    return a / b\n",
    "explanation": "Explain why a Python list comprehension can replace a simple for loop.",
    "tests": "Write pytest tests for a Python function that normalizes whitespace.",
}


@dataclass(frozen=True, slots=True)
class LocalExportResult:
    output_dir: Path
    config_path: Path
    safetensors_path: Path
    tokenizer_ref_path: Path
    generation_config_path: Path
    metadata_path: Path
    run_local_path: Path
    quantization_report_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: str(value) for key, value in payload.items()}


def export_checkpoint(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    generation_config: dict[str, Any] | None = None,
    demo_prompts: dict[str, str] | None = None,
    allow_test_tokenizer: bool = False,
) -> LocalExportResult:
    checkpoint = Path(checkpoint_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"export output already exists: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    metadata_file = metadata_path_for_checkpoint(checkpoint)
    if not metadata_file.is_file():
        raise FileNotFoundError(f"checkpoint metadata JSON is required: {metadata_file}")

    payload = torch_load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    config_data = payload.get("config")
    if not isinstance(config_data, dict):
        raise ValueError("checkpoint payload must include a model config dictionary")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint payload must include model_state_dict")

    config = ModelConfig.model_validate(config_data)
    metadata = read_metadata_json(metadata_file)
    if metadata.model_size_label != config.size_label:
        raise ValueError(
            "checkpoint metadata size does not match model config: "
            f"{metadata.model_size_label} != {config.size_label}"
        )
    if not is_starcoder2_tokenizer_id(config.tokenizer_name):
        raise ValueError("export requires a StarCoder2 tokenizer reference")
    if metadata.data_snapshot.tokenizer_id != config.tokenizer_name:
        raise ValueError(
            "data_snapshot tokenizer does not match model config tokenizer: "
            f"{metadata.data_snapshot.tokenizer_id} != {config.tokenizer_name}"
        )

    output.mkdir(parents=True, exist_ok=False)
    try:
        model = CodeWriterTutorLM(config)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        config_path = output / "config.yaml"
        safetensors_path = output / "model.safetensors"
        tokenizer_ref_path = output / "tokenizer_ref.txt"
        generation_config_path = output / "generation_config.yaml"
        export_metadata_path = output / "metadata.json"
        run_local_path = output / "run_local.py"
        quantization_report_path = output / "quantization_report.json"

        _write_yaml(config_path, config.model_dump())
        tokenizer_ref_path.write_text(config.tokenizer_name + "\n", encoding="utf-8")
        merged_generation_config = dict(DEFAULT_GENERATION_CONFIG)
        if generation_config:
            merged_generation_config.update(generation_config)
        _write_yaml(generation_config_path, merged_generation_config)
        save_model(model, safetensors_path)
        _write_json(
            export_metadata_path,
            _build_export_metadata(
                checkpoint_path=checkpoint,
                config=config,
                checkpoint_metadata=metadata.to_dict(),
                generation_config=merged_generation_config,
            ),
        )
        _write_json(quantization_report_path, build_quantization_report(config))
        run_local_path.write_text(
            build_run_local_script(allow_test_tokenizer=allow_test_tokenizer),
            encoding="utf-8",
        )
        if demo_prompts is None:
            demo_prompts = DEMO_PROMPTS
        _write_json(output / "demo_prompts.json", demo_prompts)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise

    return LocalExportResult(
        output_dir=output,
        config_path=config_path,
        safetensors_path=safetensors_path,
        tokenizer_ref_path=tokenizer_ref_path,
        generation_config_path=generation_config_path,
        metadata_path=export_metadata_path,
        run_local_path=run_local_path,
        quantization_report_path=quantization_report_path,
    )


def build_quantization_report(config: ModelConfig) -> dict[str, Any]:
    memory_report = build_local_memory_report(config).to_dict()
    return {
        "runtime_target": "cpu",
        "pytorch_dynamic_int8": {
            "status": "available_as_runtime_attempt",
            "scope": "torch.nn.Linear modules only",
            "command": "python run_local.py --quantization dynamic-int8 --prompt \"...\"",
            "caveat": (
                "Dynamic quantization must be checked per exported checkpoint for output "
                "quality and speed; it is not used during TPU training."
            ),
        },
        "bitsandbytes": {
            "status": "not_enabled_for_cpu_runtime",
            "reason": "bitsandbytes is GPU-oriented and is not relevant to the CPU laptop target.",
        },
        "custom_4bit": {
            "status": "not_implemented",
            "reason": "no verified custom 4-bit loader exists for this architecture yet.",
        },
        "gguf": {
            "status": "unsupported",
            "reason": "no verified CodeWriter-Tutor/StarCoder2 conversion path to llama.cpp GGUF.",
        },
        "memory_estimates": memory_report,
    }


def build_run_local_script(*, allow_test_tokenizer: bool = False) -> str:
    test_tokenizer_arg = (
        'parser.add_argument("--test-tokenizer", action="store_true", '
        'help="Use tiny unit-test tokenizer only for exported test artifacts.")'
        if allow_test_tokenizer
        else ""
    )
    test_tokenizer_code = (
        textwrap.indent(_TEST_TOKENIZER_CODE.strip(), "        ")
        if allow_test_tokenizer
        else ""
    )
    script = textwrap.dedent(
        f"""\
        from __future__ import annotations

        import argparse
        import json
        import sys
        import time
        from pathlib import Path
        from typing import Any

        ROOT = Path(__file__).resolve().parent
        REPO_ROOT = ROOT.parent
        if (REPO_ROOT / "cw360").is_dir() and str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        import psutil
        import torch
        import yaml
        from safetensors.torch import load_model

        from cw360.config import load_model_config
        from cw360.inference.generate import generate_text
        from cw360.model.lm import CodeWriterTutorLM
        from cw360.tokenizer.loader import ensure_pad_token_for_batching, load_tokenizer

{test_tokenizer_code}


        def main() -> None:
            parser = argparse.ArgumentParser(description="Run CodeWriter-Tutor locally on CPU.")
            parser.add_argument("--prompt", type=str, default=None)
            parser.add_argument("--prompt-file", type=Path, default=None)
            parser.add_argument(
                "--demo",
                action="store_true",
                help="Run fixed Python coding demo prompts.",
            )
            parser.add_argument("--save-output", type=Path, default=None)
            parser.add_argument("--max-new-tokens", type=int, default=None)
            parser.add_argument("--temperature", type=float, default=None)
            parser.add_argument("--top-k", type=int, default=None)
            parser.add_argument("--top-p", type=float, default=None)
            parser.add_argument("--seed", type=int, default=None)
            parser.add_argument("--no-cache", action="store_true")
            parser.add_argument(
                "--quantization",
                choices=["none", "dynamic-int8"],
                default="none",
                help="Optional CPU-only quantization attempt.",
            )
            {test_tokenizer_arg}
            args = parser.parse_args()

            try:
                generation_config = _load_generation_config(ROOT / "generation_config.yaml")
                config = load_model_config(ROOT / "config.yaml")
                model = CodeWriterTutorLM(config)
                load_model(model, ROOT / "model.safetensors", strict=False)
                model.tie_weights()
                model.to("cpu")
                model.eval()
                if args.quantization == "dynamic-int8":
                    model = torch.ao.quantization.quantize_dynamic(
                        model,
                        {{torch.nn.Linear}},
                        dtype=torch.qint8,
                    )
                    model.eval()
                tokenizer = _load_tokenizer(config.tokenizer_name, args)
                prompts = _resolve_prompts(args)
                settings = _merge_settings(generation_config, args)
                results = []
                for name, prompt in prompts.items():
                    started = time.perf_counter()
                    before = psutil.Process().memory_info().rss
                    output = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=settings["max_new_tokens"],
                        temperature=settings["temperature"],
                        top_k=settings["top_k"],
                        top_p=settings["top_p"],
                        seed=settings["seed"],
                        use_cache=settings["use_cache"],
                        device="cpu",
                    )
                    after = psutil.Process().memory_info().rss
                    result = {{
                        "name": name,
                        "prompt": prompt,
                        "output": output,
                        "seconds": time.perf_counter() - started,
                        "rss_before_bytes": before,
                        "rss_after_bytes": after,
                        "quantization": args.quantization,
                    }}
                    results.append(result)
                    print(json.dumps(result, indent=2))
                if args.save_output is not None:
                    args.save_output.parent.mkdir(parents=True, exist_ok=True)
                    args.save_output.write_text(
                        json.dumps(results, indent=2, sort_keys=True) + "\\n",
                        encoding="utf-8",
                    )
            except Exception as exc:
                raise SystemExit(_friendly_error(exc)) from exc


        def _load_generation_config(path: Path) -> dict[str, Any]:
            with path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
            if not isinstance(loaded, dict):
                raise ValueError(f"generation config must be a mapping: {{path}}")
            return loaded


        def _resolve_prompts(args: argparse.Namespace) -> dict[str, str]:
            if args.demo:
                data = json.loads((ROOT / "demo_prompts.json").read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("demo_prompts.json must contain an object")
                return {{str(key): str(value) for key, value in data.items()}}
            if args.prompt is not None:
                return {{"prompt": args.prompt}}
            if args.prompt_file is not None:
                return {{"prompt_file": args.prompt_file.read_text(encoding="utf-8")}}
            raise ValueError("provide --prompt, --prompt-file, or --demo")


        def _merge_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
            return {{
                "max_new_tokens": int(
                    args.max_new_tokens
                    if args.max_new_tokens is not None
                    else config.get("max_new_tokens", 128)
                ),
                "temperature": float(
                    args.temperature
                    if args.temperature is not None
                    else config.get("temperature", 0.0)
                ),
                "top_k": args.top_k if args.top_k is not None else config.get("top_k"),
                "top_p": args.top_p if args.top_p is not None else config.get("top_p"),
                "seed": args.seed if args.seed is not None else config.get("seed"),
                "use_cache": not args.no_cache and bool(config.get("use_cache", True)),
            }}


        def _load_tokenizer(tokenizer_name: str, args: argparse.Namespace) -> Any:
            if getattr(args, "test_tokenizer", False):
                return UnitTestTokenizer()
            tokenizer = load_tokenizer(tokenizer_name)
            ensure_pad_token_for_batching(tokenizer)
            return tokenizer


        def _friendly_error(exc: Exception) -> str:
            return (
                "Local inference failed. This runner is CPU-only and expects config.yaml, "
                "model.safetensors, generation_config.yaml, and a loadable StarCoder2 tokenizer. "
                f"Original error: {{type(exc).__name__}}: {{exc}}"
            )


        if __name__ == "__main__":
            main()
        """
    ).lstrip()
    if allow_test_tokenizer:
        script = script.replace(
            textwrap.indent(_TEST_TOKENIZER_CODE.strip(), "        "),
            _TEST_TOKENIZER_CODE.strip(),
        )
    return script


def _build_export_metadata(
    *,
    checkpoint_path: Path,
    config: ModelConfig,
    checkpoint_metadata: dict[str, Any],
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "exported_at": datetime.now(UTC).isoformat(),
        "export_format_version": 1,
        "source_checkpoint": str(checkpoint_path),
        "runtime_target": "pytorch_fp32_cpu",
        "model": {
            "model_name": config.model_name,
            "size_label": config.size_label,
            "tokenizer_name": config.tokenizer_name,
            "max_position_embeddings": config.max_position_embeddings,
        },
        "generation_config": generation_config,
        "checkpoint_metadata": checkpoint_metadata,
        "data_snapshot": checkpoint_metadata["data_snapshot"],
        "tpu_data_cursor": checkpoint_metadata.get("tpu_data_cursor"),
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "gguf_supported": False,
        "gguf_status": "unsupported_no_verified_converter",
    }


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_TEST_TOKENIZER_CODE = r'''
class UnitTestTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token = "<eos>"

    def __call__(self, prompt: str, return_tensors: str) -> dict[str, torch.Tensor]:
        del return_tensors
        values = [1]
        values.extend((ord(char) % 29) + 3 for char in prompt[:24])
        if len(values) < 2:
            values.append(self.eos_token_id)
        input_ids = torch.tensor([values], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"tok{int(token)}" for token in token_ids)
'''
