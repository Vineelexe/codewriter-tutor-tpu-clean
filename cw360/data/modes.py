from __future__ import annotations

from enum import Enum


class DataPipelineMode(str, Enum):
    TINY_LOCAL_SYNTHETIC = "tiny_local_synthetic"
    PREVIEW_REAL_SOURCES = "preview_real_sources"
    CANDIDATE_STREAM = "candidate_stream"
    EVAL_STREAM = "eval_stream"
    PREPACK_STREAM = "prepack_stream"


LOCAL_OR_PREVIEW_MODES = frozenset(
    {
        DataPipelineMode.TINY_LOCAL_SYNTHETIC,
        DataPipelineMode.PREVIEW_REAL_SOURCES,
        DataPipelineMode.CANDIDATE_STREAM,
        DataPipelineMode.EVAL_STREAM,
        DataPipelineMode.PREPACK_STREAM,
    }
)

REAL_TPU_TRAINING_INPUT = "prepacked_token_shards"


def parse_pipeline_mode(value: str | DataPipelineMode) -> DataPipelineMode:
    if isinstance(value, DataPipelineMode):
        return value
    try:
        return DataPipelineMode(value)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in DataPipelineMode)
        raise ValueError(
            f"unsupported data pipeline mode {value!r}; expected one of {allowed}"
        ) from exc


def mode_allows_live_streaming(mode: str | DataPipelineMode) -> bool:
    parsed = parse_pipeline_mode(mode)
    return parsed in LOCAL_OR_PREVIEW_MODES


def mode_is_tpu_training_path(mode: str | DataPipelineMode) -> bool:
    parse_pipeline_mode(mode)
    return False
