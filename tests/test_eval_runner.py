from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from cw360.config import ModelConfig
from cw360.eval.runner import EvalRunConfig, run_evaluation
from cw360.model.lm import CodeWriterTutorLM


class TinyTokenizer:
    eos_token_id = 0
    pad_token = "<pad>"

    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        token_count = max(1, min(4, len(text.split())))
        input_ids = torch.arange(1, token_count + 1, dtype=torch.long).unsqueeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def decode(self, token_ids: object, **_: object) -> str:
        if isinstance(token_ids, torch.Tensor):
            values = token_ids.tolist()
        else:
            values = list(token_ids)  # type: ignore[arg-type]
        return " ".join(f"tok{value}" for value in values)



def tiny_config() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "config_type": "model",
            "size_label": "tiny",
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
        }
    )


def write_model_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "config_type: model",
                "size_label: tiny",
                "tokenizer_name: bigcode/starcoder2-15b",
                "vocab_size: 16",
                "max_position_embeddings: 32",
                "hidden_size: 16",
                "num_hidden_layers: 1",
                "num_attention_heads: 4",
                "num_key_value_heads: 2",
                "intermediate_size: 32",
            ]
        ),
        encoding="utf-8",
    )


def test_run_evaluation_writes_jsonl_and_markdown_with_required_metadata(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "category": "python_debugging",
                "prompt": "Fix average([]).",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    model_config_path = tmp_path / "model.yaml"
    write_model_config(model_config_path)
    model = CodeWriterTutorLM(tiny_config(), init_seed=1)

    config = EvalRunConfig(
        model_config_path=model_config_path,
        prompt_paths=(prompt_path,),
        output_dir=tmp_path / "eval",
        checkpoint_name="random_init",
        step=0,
        max_new_tokens=2,
        notes="test run",
        training_stage="eval_only",
        data_snapshot={"snapshot": "unit-test"},
        use_cache=False,
    )
    jsonl_path, markdown_path = run_evaluation(config, model=model, tokenizer=TinyTokenizer())

    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["prompt"] == "Fix average([])."
    assert record["checkpoint_name"] == "random_init"
    assert record["step"] == 0
    assert record["training_stage"] == "eval_only"
    assert record["model_size_label"] == "tiny"
    assert record["data_snapshot"] == {"snapshot": "unit-test"}
    assert {item["name"] for item in record["scoring_schema"]} == {
        "code_correctness",
        "explanation_quality",
        "instruction_following",
        "safety_scope",
    }
    report = markdown_path.read_text(encoding="utf-8")
    assert "Evaluation Report: random_init" in report
    assert "Fix average([])." in report
