from __future__ import annotations

import itertools
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from cw360.checkpoint import (
    CheckpointMetadata,
    DataSnapshot,
    TPUDataCursor,
    get_git_commit,
    load_checkpoint,
    save_checkpoint,
)
from cw360.config import load_config, load_model_config
from cw360.model import CodeWriterTutorLM
from cw360.prepack.manifest import hash_payload
from cw360.train.metrics import MetricsTracker
from cw360.train.optim import AdamWConfig, build_adamw
from cw360.train.precision import PrecisionConfig, autocast_context
from cw360.train.schedulers import SchedulerConfig, WarmupStableDecayScheduler
from cw360.train.state import TrainingState
from cw360.train.trainer import (
    TrainingBatch,
    TrainingRunResult,
    compute_causal_lm_loss,
    make_synthetic_causal_batch,
)
from cw360.train.validate import validate_stage_for_training, validate_tiny_cpu_model

BatchFactory = Callable[[TPUDataCursor | None], Iterator[TrainingBatch]]


class CPUTinyTrainer:
    def __init__(
        self,
        *,
        config: Mapping[str, Any] | str | Path,
        output_dir: str | Path,
        batch_factory: BatchFactory | None = None,
        val_batch_factory: BatchFactory | None = None,
        data_snapshot: DataSnapshot | None = None,
        resume_checkpoint: str | Path | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = _load_train_config(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_stage = str(self.config["training_stage"])
        validate_stage_for_training(self.training_stage)

        self.model_config = load_model_config(self.config["model_config_path"])
        validate_tiny_cpu_model(self.model_config)
        self.device = torch.device("cpu")
        self.model = CodeWriterTutorLM(
            self.model_config,
            init_seed=int(self.config.get("init_seed", 1234)),
        ).to(self.device)
        self.optimizer = build_adamw(
            self.model.parameters(),
            AdamWConfig(
                learning_rate=float(self.config["learning_rate"]),
                weight_decay=float(self.config.get("weight_decay", 0.1)),
                beta1=float(self.config.get("adam_beta1", 0.9)),
                beta2=float(self.config.get("adam_beta2", 0.95)),
                eps=float(self.config.get("adam_eps", 1.0e-8)),
            ),
        )
        total_steps = max(1, int(self.config.get("max_steps", 1)))
        self.scheduler = WarmupStableDecayScheduler(
            self.optimizer,
            SchedulerConfig(
                total_steps=total_steps,
                warmup_steps=int(self.config.get("warmup_steps", 0)),
                stable_steps=_optional_int(self.config.get("stable_steps")),
                min_lr_ratio=float(self.config.get("min_lr_ratio", 0.1)),
                schedule_type=str(self.config.get("schedule_type", "wsd")),
            ),
        )
        self.precision = PrecisionConfig(precision="float32", backend="cpu")
        self.gradient_accumulation_steps = int(
            self.config.get("gradient_accumulation_steps", 1)
        )
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.metrics = MetricsTracker()
        self.state = TrainingState()
        self.resume_checkpoint = Path(resume_checkpoint) if resume_checkpoint is not None else None
        self.previous_checkpoint: str | None = None
        self.data_cursor: TPUDataCursor | None = None
        self.run_id = run_id or str(self.config.get("run_id") or uuid.uuid4())
        self.data_snapshot = data_snapshot or synthetic_data_snapshot(self.config)
        self.batch_factory = batch_factory or self._synthetic_batch_factory
        self.val_batch_factory = val_batch_factory

        if self.resume_checkpoint is not None:
            metadata, cursor = load_checkpoint(
                checkpoint_path=self.resume_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                expected_model_size_label="tiny",
                expected_training_stage=self.training_stage,
                expected_data_snapshot=data_snapshot,
            )
            self.state = TrainingState.from_checkpoint_values(
                step=metadata.step,
                tokens_seen=metadata.tokens_seen,
                sequences_seen=metadata.sequences_seen,
            )
            self.previous_checkpoint = str(self.resume_checkpoint)
            self.data_cursor = cursor
            self.run_id = metadata.run_id

    def train(self, *, target_steps: int | None = None) -> TrainingRunResult:
        max_steps = int(target_steps if target_steps is not None else self.config["max_steps"])
        if max_steps < self.state.step:
            raise ValueError("target max_steps is behind checkpoint step")
        checkpoint_paths: list[Path] = []
        last_loss = float("nan")
        batches = self.batch_factory(self.data_cursor)

        self.model.train()
        while self.state.step < max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            step_tokens = 0
            step_sequences = 0
            step_loss = 0.0
            last_cursor: TPUDataCursor | None = None
            step_timer = _StepTimer()
            for _ in range(self.gradient_accumulation_steps):
                try:
                    batch = next(batches)
                except StopIteration:
                    batches = self.batch_factory(None)
                    batch = next(batches)
                batch = self._prepare_batch(batch)
                with autocast_context(self.precision):
                    output = self.model(batch.input_ids)
                    loss = compute_causal_lm_loss(output.logits, batch.labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss: {loss.item()}")
                (loss / self.gradient_accumulation_steps).backward()
                step_loss += float(loss.detach().item())
                step_tokens += batch.num_tokens
                step_sequences += batch.num_sequences
                last_cursor = batch.tpu_data_cursor

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.state.advance(tokens=step_tokens, sequences=step_sequences)
            if last_cursor is not None:
                self.data_cursor = self._progress_cursor(last_cursor)
            last_loss = step_loss / self.gradient_accumulation_steps
            metric = self.metrics.record_step(
                loss=last_loss,
                learning_rate=self.scheduler.get_last_lr()[0],
                step=self.state.step,
                tokens_seen=self.state.tokens_seen,
                sequences_seen=self.state.sequences_seen,
                step_tokens=step_tokens,
                step_sequences=step_sequences,
                step_time=step_timer.elapsed(),
            )
            if bool(self.config.get("log_steps", False)):
                print(
                    f"step={metric.step} loss={metric.loss:.4f} "
                    f"lr={metric.learning_rate:.6g} tok/s={metric.tokens_per_second:.2f}"
                )

        checkpoint_paths.append(self.save_checkpoint(train_loss=last_loss))
        return TrainingRunResult(
            state=replace(self.state),
            checkpoints=tuple(checkpoint_paths),
            metrics=tuple(self.metrics.history),
            final_loss=last_loss,
            resumed_from=self.resume_checkpoint,
        )

    def save_checkpoint(self, *, train_loss: float) -> Path:
        val_loss = self.validation_loss()
        metadata = CheckpointMetadata(
            model_name=self.model_config.model_name,
            model_size_label=self.model_config.size_label,
            run_id=self.run_id,
            step=self.state.step,
            train_loss=float(train_loss),
            val_loss=val_loss,
            tokens_seen=self.state.tokens_seen,
            sequences_seen=self.state.sequences_seen,
            trained_by=str(self.config.get("trained_by", "vineel")),
            date=datetime.now(tz=UTC).date().isoformat(),
            platform=str(self.config.get("platform", "local-cpu-test")),
            training_stage=self.training_stage,
            previous_checkpoint=self.previous_checkpoint,
            dataset_mixture=dict(self.config.get("dataset_mixture", {"tiny_synthetic": 1.0})),
            context_length=int(self.config["max_seq_len"]),
            optimizer="adamw",
            learning_rate=float(self.scheduler.get_last_lr()[0]),
            batch_size=int(self.config["micro_batch_size"]),
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            precision=self.precision.precision,
            backend="cpu",
            git_commit=get_git_commit(),
            notes=str(self.config.get("notes", "phase9 tiny CPU training")),
            data_snapshot=self.data_snapshot,
            tpu_data_cursor=self.data_cursor,
        )
        path = save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            output_dir=self.output_dir,
            metadata=metadata,
        )
        if path is None:
            raise RuntimeError("CPU checkpoint save unexpectedly returned None")
        self.previous_checkpoint = str(path)
        return path

    def validation_loss(self) -> float | None:
        if self.val_batch_factory is None:
            return None
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in itertools.islice(self.val_batch_factory(None), int(self.config.get("val_batches", 1))):
                batch = self._prepare_batch(batch)
                output = self.model(batch.input_ids)
                losses.append(float(compute_causal_lm_loss(output.logits, batch.labels).item()))
        self.model.train()
        if not losses:
            return None
        return sum(losses) / len(losses)

    def _prepare_batch(self, batch: TrainingBatch) -> TrainingBatch:
        local_seq_len = _optional_int(self.config.get("local_cpu_train_seq_len"))
        if local_seq_len is not None:
            batch = batch.crop(min(local_seq_len, self.model_config.max_position_embeddings))
        if batch.input_ids.shape != batch.labels.shape:
            raise ValueError("input_ids and labels must have the same shape")
        if batch.input_ids.ndim != 2:
            raise ValueError("training batches must have shape [batch, seq_len]")
        if int(batch.input_ids.shape[1]) > self.model_config.max_position_embeddings:
            raise ValueError("batch seq_len exceeds model max_position_embeddings")
        return batch.to(self.device)

    def _synthetic_batch_factory(self, cursor: TPUDataCursor | None) -> Iterator[TrainingBatch]:
        del cursor
        batch_size = int(self.config["micro_batch_size"])
        seq_len = int(
            self.config.get(
                "local_cpu_train_seq_len",
                min(int(self.config["max_seq_len"]), self.model_config.max_position_embeddings),
            )
        )
        seed = int(self.config.get("seed", 1234))
        step = 0
        while True:
            yield make_synthetic_causal_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=self.model_config.vocab_size,
                seed=seed + self.state.step + step,
            )
            step += 1

    def _progress_cursor(self, cursor: TPUDataCursor) -> TPUDataCursor:
        return TPUDataCursor(
            epoch=cursor.epoch,
            global_step=self.state.step,
            tokens_seen=self.state.tokens_seen,
            sequences_seen=self.state.sequences_seen,
            shard_order_seed=cursor.shard_order_seed,
            shard_index=cursor.shard_index,
            sequence_offset=cursor.sequence_offset,
            consumed_sequences_in_current_shard=cursor.consumed_sequences_in_current_shard,
            drop_last=cursor.drop_last,
            batch_size_per_device=cursor.batch_size_per_device,
            num_devices=cursor.num_devices,
            resume_policy=cursor.resume_policy,
        )


def synthetic_data_snapshot(config: Mapping[str, Any]) -> DataSnapshot:
    max_seq_len = int(config.get("max_seq_len", 32))
    snapshot_hash = hash_payload(
        {
            "source": "tiny_synthetic_cpu",
            "stage": str(config.get("training_stage", "base_pretrain")),
            "max_seq_len": max_seq_len,
        }
    )
    return DataSnapshot(
        datasets_config_hash=snapshot_hash,
        mixture_config_hash=snapshot_hash,
        synthetic_repo_id="local-tiny-synthetic",
        synthetic_shard_list=["synthetic://tiny-cpu"],
        teacher_manifest_id=None,
        candidate_manifest_id=None,
        split_manifest_id="tiny-cpu-split",
        tokenizer_id="bigcode/starcoder2-15b",
        tokenizer_config_hash="tiny-cpu-tokenizer",
        prepacked_dataset_name="tiny-cpu-synthetic",
        prepacked_dataset_version="phase9",
        prepacked_manifest_hash=snapshot_hash,
        shard_manifest_hash=snapshot_hash,
        shard_file_list=["synthetic://tiny-cpu"],
        shard_hashes={"synthetic://tiny-cpu": snapshot_hash},
        seq_len_plus_one=max_seq_len + 1,
        token_dtype="uint16",
    )


def _load_train_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, str | Path):
        return load_config(config)
    return dict(config)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


class _StepTimer:
    def __init__(self) -> None:
        self.started_at = datetime.now(tz=UTC)

    def elapsed(self) -> float:
        delta = datetime.now(tz=UTC) - self.started_at
        return max(delta.total_seconds(), 1.0e-12)
