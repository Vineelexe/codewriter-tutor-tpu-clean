from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from cw360.checkpoint import (
    CheckpointMetadata,
    TPUDataCursor,
    get_git_commit,
    load_checkpoint,
    save_checkpoint,
)
from cw360.config import load_config, load_model_config
from cw360.model import CodeWriterTutorLM
from cw360.train.metrics import MetricsTracker
from cw360.train.optim import AdamWConfig, build_adamw
from cw360.train.precision import PrecisionConfig, autocast_context
from cw360.train.schedulers import SchedulerConfig, WarmupStableDecayScheduler
from cw360.train.shard_dataloader import (
    PrepackedShardDataLoader,
    data_snapshot_from_prepacked_manifest,
    validate_prepacked_manifest_for_training,
)
from cw360.train.state import TrainingState
from cw360.train.time_guard import PeriodicSaveTimer, SaveBeforeExitTimeGuard
from cw360.train.trainer import TrainingBatch, TrainingRunResult, compute_causal_lm_loss
from cw360.train.validate import validate_stage_for_training, validate_tpu_train_config
from cw360.train.xla_utils import (
    TorchXLANotAvailable,
    import_xla_model,
    is_xla_main_process,
    xla_device,
    xla_optimizer_step,
)

CheckpointCallback = Callable[[Path], None]


class TPUTrainer:
    """TPU/XLA entrypoint wrapper.

    The implementation intentionally keeps torch_xla imports inside train() so
    CPU tests and local imports do not require TPU dependencies.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | str | Path,
        prepacked_dir: str | Path,
        output_dir: str | Path,
        resume_checkpoint: str | Path | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = load_config(config) if isinstance(config, str | Path) else dict(config)
        validate_tpu_train_config(config if isinstance(config, str | Path) else self.config)
        self.prepacked_dir = Path(prepacked_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume_checkpoint = None if resume_checkpoint is None else Path(resume_checkpoint)
        self.checkpoint_callback = checkpoint_callback
        self.run_id = run_id or str(self.config.get("run_id") or uuid.uuid4())
        self.training_stage = str(self.config["training_stage"])
        validate_stage_for_training(self.training_stage)
        self.model_config = load_model_config(self.config["model_config_path"])
        self.data_snapshot = data_snapshot_from_prepacked_manifest(self.prepacked_dir)
        self.state = TrainingState()
        self.previous_checkpoint: str | None = None
        self.data_cursor: TPUDataCursor | None = None
        self.saved_steps: set[int] = set()

    def train(self, *, target_steps: int | None = None) -> TrainingRunResult:
        try:
            xm = import_xla_model()
            device = xla_device()
        except TorchXLANotAvailable:
            raise
        del xm

        manifest = PrepackedShardDataLoader(
            self.prepacked_dir,
            split="train",
            batch_size=int(self.config["micro_batch_size"]),
            drop_last=bool(self.config.get("drop_last", True)),
            static_shapes=bool(self.config.get("static_shapes", True)),
            shard_order_seed=int(self.config.get("shard_order_seed", 0)),
            num_devices=_xla_world_size(default=1),
        ).manifest
        validate_prepacked_manifest_for_training(manifest)

        model = CodeWriterTutorLM(
            self.model_config,
            init_seed=int(self.config.get("init_seed", 1234)),
        ).to(device)
        optimizer = build_adamw(
            model.parameters(),
            AdamWConfig(
                learning_rate=float(self.config["learning_rate"]),
                weight_decay=float(self.config.get("weight_decay", 0.1)),
                beta1=float(self.config.get("adam_beta1", 0.9)),
                beta2=float(self.config.get("adam_beta2", 0.95)),
                eps=float(self.config.get("adam_eps", 1.0e-8)),
            ),
        )
        max_steps = int(target_steps if target_steps is not None else self.config["max_steps"])
        scheduler = WarmupStableDecayScheduler(
            optimizer,
            SchedulerConfig(
                total_steps=max(1, max_steps),
                warmup_steps=int(self.config.get("warmup_steps", 0)),
                stable_steps=_optional_int(self.config.get("stable_steps")),
                min_lr_ratio=float(self.config.get("min_lr_ratio", 0.1)),
                schedule_type=str(self.config.get("schedule_type", "wsd")),
            ),
        )
        if self.resume_checkpoint is not None:
            metadata, cursor = load_checkpoint(
                checkpoint_path=self.resume_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_model_size_label=self.model_config.size_label,
                expected_training_stage=self.training_stage,
                expected_data_snapshot=self.data_snapshot,
                map_location="cpu",
            )
            self.state = TrainingState.from_checkpoint_values(
                step=metadata.step,
                tokens_seen=metadata.tokens_seen,
                sequences_seen=metadata.sequences_seen,
            )
            self.previous_checkpoint = str(self.resume_checkpoint)
            self.data_cursor = cursor
            self.run_id = metadata.run_id

        if max_steps < self.state.step:
            raise ValueError("target max_steps is behind checkpoint step")

        metrics = MetricsTracker()
        precision = PrecisionConfig(
            precision=str(self.config.get("precision", "bf16")),  # type: ignore[arg-type]
            backend="xla_tpu",
        )
        accumulation_steps = int(self.config.get("gradient_accumulation_steps", 1))
        if accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        save_interval_minutes = self.config.get(
            "save_every_n_minutes",
            self.config.get("checkpoint_interval_minutes", 30),
        )
        periodic_save = PeriodicSaveTimer(interval_seconds=60.0 * float(save_interval_minutes))
        save_before_exit_minutes = self.config.get(
            "save_before_exit_minutes",
            self.config.get("save_before_exit_margin_minutes", 20),
        )
        exit_guard = SaveBeforeExitTimeGuard(
            max_runtime_seconds=60.0 * float(self.config.get("session_time_limit_minutes", 510)),
            save_margin_seconds=60.0 * float(save_before_exit_minutes),
        )
        checkpoint_paths: list[Path] = []
        last_loss = float("nan")
        batches = self._batch_iterator(self.data_cursor)
        model.train()

        while self.state.step < max_steps:
            optimizer.zero_grad(set_to_none=True)
            step_tokens = 0
            step_sequences = 0
            step_loss = 0.0
            last_cursor: TPUDataCursor | None = None
            started = datetime.now(tz=UTC)
            for _ in range(accumulation_steps):
                try:
                    batch = next(batches)
                except StopIteration:
                    batches = self._batch_iterator(None)
                    batch = next(batches)
                batch = batch.to(device)
                with autocast_context(precision):
                    output = model(batch.input_ids)
                    loss = compute_causal_lm_loss(output.logits, batch.labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss: {loss.item()}")
                (loss / accumulation_steps).backward()
                step_loss += float(loss.detach().cpu().item())
                step_tokens += batch.num_tokens
                step_sequences += batch.num_sequences
                last_cursor = batch.tpu_data_cursor

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            xla_optimizer_step(optimizer)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            self.state.advance(tokens=step_tokens, sequences=step_sequences)
            if last_cursor is not None:
                self.data_cursor = self._progress_cursor(last_cursor)
            last_loss = step_loss / accumulation_steps
            metric = metrics.record_step(
                loss=last_loss,
                learning_rate=scheduler.get_last_lr()[0],
                step=self.state.step,
                tokens_seen=self.state.tokens_seen,
                sequences_seen=self.state.sequences_seen,
                step_tokens=step_tokens,
                step_sequences=step_sequences,
                step_time=max((datetime.now(tz=UTC) - started).total_seconds(), 1.0e-12),
            )
            if bool(self.config.get("log_steps", False)) and is_xla_main_process():
                print(
                    f"step={metric.step} loss={metric.loss:.4f} "
                    f"lr={metric.learning_rate:.6g} tok/s={metric.tokens_per_second:.2f}"
                )

            if periodic_save.should_save() or exit_guard.should_save_before_exit():
                saved = self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    precision=precision,
                    train_loss=last_loss,
                )
                if saved is not None:
                    checkpoint_paths.append(saved)
                periodic_save.mark_saved()
                if exit_guard.remaining_seconds() <= 0:
                    break

        saved = self.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            precision=precision,
            train_loss=last_loss,
        )
        if saved is not None:
            checkpoint_paths.append(saved)
        return TrainingRunResult(
            state=replace(self.state),
            checkpoints=tuple(checkpoint_paths),
            metrics=tuple(metrics.history),
            final_loss=last_loss,
            resumed_from=self.resume_checkpoint,
        )

    def save_checkpoint(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        precision: PrecisionConfig,
        train_loss: float,
    ) -> Path | None:
        if self.state.step in self.saved_steps:
            return None
        metadata = CheckpointMetadata(
            model_name=self.model_config.model_name,
            model_size_label=self.model_config.size_label,
            run_id=self.run_id,
            step=self.state.step,
            train_loss=float(train_loss),
            val_loss=None,
            tokens_seen=self.state.tokens_seen,
            sequences_seen=self.state.sequences_seen,
            trained_by=str(
                self.config.get("trainer_name", self.config.get("trained_by", "vineel"))
            ),
            date=datetime.now(tz=UTC).date().isoformat(),
            platform=str(self.config.get("platform", "kaggle-tpu-background")),
            training_stage=self.training_stage,
            previous_checkpoint=self.previous_checkpoint,
            dataset_mixture=dict(self.config.get("dataset_mixture", {"prepacked": 1.0})),
            context_length=int(self.config["max_seq_len"]),
            optimizer="adamw",
            learning_rate=float(scheduler.get_last_lr()[0]),
            batch_size=int(self.config["micro_batch_size"]),
            gradient_accumulation_steps=int(self.config.get("gradient_accumulation_steps", 1)),
            precision=precision.precision,
            backend="xla_tpu",
            git_commit=get_git_commit(),
            notes=str(self.config.get("notes", "phase10 Kaggle TPU relay training")),
            data_snapshot=self.data_snapshot,
            tpu_data_cursor=self.data_cursor,
        )
        path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=self.output_dir,
            metadata=metadata,
            config=self.model_config.model_dump(),
        )
        self.saved_steps.add(self.state.step)
        if path is not None:
            self.previous_checkpoint = str(path)
            if self.checkpoint_callback is not None:
                self.checkpoint_callback(path)
        return path

    def _batch_iterator(self, cursor: TPUDataCursor | None) -> Iterator[TrainingBatch]:
        return iter(
            PrepackedShardDataLoader(
                self.prepacked_dir,
                split="train",
                batch_size=int(self.config["micro_batch_size"]),
                drop_last=bool(self.config.get("drop_last", True)),
                static_shapes=bool(self.config.get("static_shapes", True)),
                shard_order_seed=int(self.config.get("shard_order_seed", 0)),
                cursor=cursor,
                num_devices=_xla_world_size(default=1),
            )
        )

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


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _xla_world_size(*, default: int | None) -> int | None:
    try:
        xm = import_xla_model()
        return int(xm.xrt_world_size())
    except Exception:
        return default
