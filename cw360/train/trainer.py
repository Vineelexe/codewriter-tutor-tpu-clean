from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from cw360.checkpoint.data_cursor import TPUDataCursor
from cw360.train.metrics import StepMetrics
from cw360.train.state import TrainingState


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    tpu_data_cursor: TPUDataCursor | None = None
    data_wait_time: float = 0.0

    def to(self, device: torch.device | str) -> "TrainingBatch":
        return TrainingBatch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            tpu_data_cursor=self.tpu_data_cursor,
            data_wait_time=self.data_wait_time,
        )

    def crop(self, seq_len: int | None) -> "TrainingBatch":
        if seq_len is None:
            return self
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        return TrainingBatch(
            input_ids=self.input_ids[:, :seq_len],
            labels=self.labels[:, :seq_len],
            tpu_data_cursor=self.tpu_data_cursor,
            data_wait_time=self.data_wait_time,
        )

    @property
    def num_sequences(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.labels.numel())


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    state: TrainingState
    checkpoints: tuple[Path, ...]
    metrics: tuple[StepMetrics, ...]
    final_loss: float
    resumed_from: Path | None = None


def compute_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq_len, vocab_size]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [batch, seq_len]")
    return F.cross_entropy(
        logits.contiguous().view(-1, logits.shape[-1]),
        labels.contiguous().view(-1),
        ignore_index=-100,
    )


def make_synthetic_causal_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> TrainingBatch:
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than 1")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sequence = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len + 1),
        dtype=torch.long,
        generator=generator,
    )
    return TrainingBatch(input_ids=sequence[:, :-1], labels=sequence[:, 1:])


def repeat_batch(batch: TrainingBatch) -> Iterator[TrainingBatch]:
    while True:
        yield batch


def cycle_finite_batches(batches: Iterable[TrainingBatch]) -> Iterator[TrainingBatch]:
    cached = tuple(batches)
    if not cached:
        raise ValueError("at least one batch is required")
    while True:
        yield from cached
