from __future__ import annotations

from cw360.data.batching import build_causal_lm_batch
from cw360.data.packing import continuous_pack
from cw360.data.schemas import TrainingExample
from cw360.data.tokenize import tokenize_example


class TinyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) for char in text]


def test_tokenize_appends_single_eos_and_preserves_code_text() -> None:
    tokenizer = TinyTokenizer()
    example = TrainingExample(
        text="def add(a, b):\n    return a + b\n",
        source="unit",
        example_type="python_code",
        metadata={"id": "ex-1"},
        loss_weight=0.75,
    )

    tokenized = tokenize_example(example, tokenizer, preserve_text=True)

    assert tokenized.token_ids[-1] == tokenizer.eos_token_id
    assert tokenized.text == example.text
    assert "    return" in tokenized.text
    assert tokenized.metadata == {"id": "ex-1"}
    assert tokenized.loss_weight == 0.75

    tokenized_again = tokenize_example(
        TrainingExample(text="x" + chr(0), source="unit", example_type="python_code"),
        tokenizer,
    )
    assert tokenized_again.token_ids[-1] == tokenizer.eos_token_id
    assert tokenized_again.token_ids.count(tokenizer.eos_token_id) == 1


def test_continuous_pack_fills_sequence_and_labels_shift() -> None:
    sequences = continuous_pack(
        [[1, 2, 0], [3, 0]],
        max_seq_len=4,
    )

    assert [sequence.token_ids for sequence in sequences] == [[1, 2, 0, 3, 0]]

    batch = build_causal_lm_batch(sequences, seq_len=4, pad_token_id=0)
    assert batch.input_ids.tolist() == [[1, 2, 0, 3]]
    assert batch.labels.tolist() == [[2, 0, 3, 0]]
    assert batch.input_ids.shape == (1, 4)
    assert batch.labels.shape == (1, 4)
