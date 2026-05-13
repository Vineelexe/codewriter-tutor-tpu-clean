from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class BoundedBuffer(Generic[T]):
    max_size: int
    _items: deque[T] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def append(self, item: T) -> None:
        if len(self._items) >= self.max_size:
            raise OverflowError("bounded buffer is full")
        self._items.append(item)

    def extend(self, items: Iterable[T]) -> None:
        for item in items:
            self.append(item)

    def popleft(self) -> T:
        return self._items.popleft()

    def drain(self, count: int | None = None) -> list[T]:
        if count is not None and count < 0:
            raise ValueError("count must be non-negative")
        limit = len(self._items) if count is None else min(count, len(self._items))
        return [self._items.popleft() for _ in range(limit)]

    @property
    def full(self) -> bool:
        return len(self._items) >= self.max_size

    @property
    def empty(self) -> bool:
        return not self._items
