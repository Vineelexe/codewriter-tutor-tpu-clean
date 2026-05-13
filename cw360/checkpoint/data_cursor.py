from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class TPUDataCursor:
    epoch: int
    global_step: int
    tokens_seen: int
    sequences_seen: int
    shard_order_seed: int
    shard_index: int
    sequence_offset: int
    consumed_sequences_in_current_shard: int
    drop_last: bool
    batch_size_per_device: int
    num_devices: int | None
    resume_policy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TPUDataCursor":
        return cls(
            epoch=int(data["epoch"]),
            global_step=int(data["global_step"]),
            tokens_seen=int(data["tokens_seen"]),
            sequences_seen=int(data["sequences_seen"]),
            shard_order_seed=int(data["shard_order_seed"]),
            shard_index=int(data["shard_index"]),
            sequence_offset=int(data["sequence_offset"]),
            consumed_sequences_in_current_shard=int(data["consumed_sequences_in_current_shard"]),
            drop_last=bool(data["drop_last"]),
            batch_size_per_device=int(data["batch_size_per_device"]),
            num_devices=None if data.get("num_devices") is None else int(data["num_devices"]),
            resume_policy=str(data["resume_policy"]),
        )


Backend = Literal["cpu", "cuda", "xla_tpu"]
