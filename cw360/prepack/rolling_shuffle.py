from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


def rolling_shuffle(
    items: Iterable[T],
    *,
    buffer_size: int,
    seed: int,
) -> Iterator[T]:
    sink: list[T] = []

    def emit(item: T) -> None:
        sink.append(item)

    shuffler = RollingShuffleBuffer(buffer_size=buffer_size, seed=seed, emit=emit)
    for item in items:
        shuffler.add(item)
        while sink:
            yield sink.pop(0)
    shuffler.close()
    while sink:
        yield sink.pop(0)


@dataclass(slots=True)
class RollingShuffleBuffer(Generic[T]):
    buffer_size: int
    seed: int
    emit: Callable[[T], None]
    _buffer: list[T] = field(default_factory=list, init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.buffer_size <= 0:
            raise ValueError("rolling shuffle buffer_size must be positive")
        self._rng = random.Random(self.seed)

    def add(self, item: T) -> None:
        if len(self._buffer) < self.buffer_size:
            self._buffer.append(item)
            return
        index = self._rng.randrange(len(self._buffer))
        outgoing = self._buffer[index]
        self._buffer[index] = item
        self.emit(outgoing)

    def close(self) -> None:
        self._rng.shuffle(self._buffer)
        for item in self._buffer:
            self.emit(item)
        self._buffer = []


def resolve_shuffle_buffer_sequences(
    *,
    configured_sequences: int,
    minimum_sequences: int,
    seq_len_plus_one: int,
    max_ram_fraction: float,
    memory_guard: bool,
) -> int:
    if configured_sequences <= 0:
        raise ValueError("configured shuffle buffer must be positive")
    if not memory_guard:
        return configured_sequences
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except Exception:
        return max(1, min(configured_sequences, minimum_sequences))

    bytes_per_sequence = seq_len_plus_one * 2
    guarded = int((available * max_ram_fraction) // bytes_per_sequence)
    if guarded <= 0:
        guarded = 1
    return max(1, min(configured_sequences, max(minimum_sequences, guarded)))
