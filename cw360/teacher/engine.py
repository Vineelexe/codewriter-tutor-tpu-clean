from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cw360.config import load_yaml
from cw360.teacher.cache import TeacherResponseCache
from cw360.teacher.manifest import (
    TeacherRunManifest,
    load_manifest,
    manifest_path,
    save_manifest,
)
from cw360.teacher.normalization import normalize_teacher_payload
from cw360.teacher.providers import (
    BaseTeacherClient,
    GeminiTeacherClient,
    MockTeacherClient,
    NvidiaNIMTeacherClient,
    OpenAICompatibleTeacherClient,
)
from cw360.teacher.queue import TeacherRequest, build_request_queue, load_candidate_inputs
from cw360.teacher.rate_limit import RateLimitConfig, RateLimiter
from cw360.teacher.response_validation import parse_json_response, validate_response_payload
from cw360.teacher.retry import RetryConfig, run_with_retries
from cw360.teacher.writer import StructuredShardWriter


@dataclass(slots=True)
class TeacherEngineConfig:
    run_id: str
    provider: str = "mock"
    model: str = "mock-teacher"
    max_requests_per_run: int = 2000
    max_concurrency: int = 4
    max_retries: int = 5
    timeout_seconds: int = 60
    temperature: float = 0.2
    max_output_tokens: int = 1200
    dry_run: bool = False
    estimate_only: bool = False
    storage: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def local_temp_dir(self) -> Path:
        return Path(str(self.storage.get("local_temp_dir", "/tmp/cw360_teacher")))

    @property
    def shard_size(self) -> int:
        return int(self.storage.get("shard_size", 1000))

    @property
    def candidate_shards(self) -> list[str]:
        return [str(path) for path in self.inputs.get("candidate_shards", [])]


@dataclass(slots=True)
class TeacherEngineResult:
    planned_requests: int
    completed: int
    failed: int
    output_dir: str
    manifest_path: str
    dry_run: bool = False


class TeacherGenerationEngine:
    def __init__(
        self,
        config: TeacherEngineConfig,
        *,
        client: BaseTeacherClient | None = None,
        local_only: bool = True,
    ) -> None:
        self.config = config
        self.client = client or create_teacher_client(config)
        self.local_only = local_only
        self.output_dir = config.local_temp_dir / config.run_id
        self.manifest_file = manifest_path(self.output_dir, config.run_id)
        self.manifest = load_manifest(self.manifest_file) or TeacherRunManifest(
            run_id=config.run_id,
            provider=config.provider,
            model=config.model,
            output_dir=str(self.output_dir),
            dry_run=config.dry_run,
        )
        self.cache = TeacherResponseCache(self.output_dir / "response_cache.jsonl")
        self.writer = StructuredShardWriter(
            self.output_dir,
            shard_size=config.shard_size,
            starting_index=len(self.manifest.written_shards),
        )
        self.rate_limiter = RateLimiter(
            RateLimitConfig(config.storage.get("requests_per_minute"))
        )

    def estimate(self) -> TeacherEngineResult:
        requests = self._build_requests()
        return TeacherEngineResult(
            planned_requests=len(requests[: self.config.max_requests_per_run]),
            completed=len(self.manifest.completed_candidates),
            failed=len(self.manifest.failed_candidates),
            output_dir=str(self.output_dir),
            manifest_path=str(self.manifest_file),
            dry_run=True,
        )

    def run(self) -> TeacherEngineResult:
        requests = self._build_requests()[: self.config.max_requests_per_run]
        if self.config.dry_run or self.config.estimate_only:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_manifest(self.manifest, self.manifest_file)
            return TeacherEngineResult(
                planned_requests=len(requests),
                completed=len(self.manifest.completed_candidates),
                failed=len(self.manifest.failed_candidates),
                output_dir=str(self.output_dir),
                manifest_path=str(self.manifest_file),
                dry_run=True,
            )

        self.client.validate_environment()
        for request in requests:
            self._process_request(request)
            save_manifest(self.manifest, self.manifest_file)
        shard = self.writer.flush()
        if shard.count:
            self.manifest.written_shards.extend(self.writer.written_manifest_rows())
        save_manifest(self.manifest, self.manifest_file)
        self.client.close()
        return TeacherEngineResult(
            planned_requests=len(requests),
            completed=len(self.manifest.completed_candidates),
            failed=len(self.manifest.failed_candidates),
            output_dir=str(self.output_dir),
            manifest_path=str(self.manifest_file),
            dry_run=False,
        )

    def _build_requests(self) -> list[TeacherRequest]:
        candidates = load_candidate_inputs(self.config.candidate_shards)
        return build_request_queue(
            candidates,
            completed_candidate_ids=set(self.manifest.completed_candidates),
        )

    def _process_request(self, request: TeacherRequest) -> None:
        cached = self.cache.get(request.request_hash)
        if cached is not None:
            normalized_payload = self._normalize_cached_payload(cached, request)
            validation = validate_response_payload(
                normalized_payload,
                source_prompt=request.prompt,
            )
            if validation.ok:
                self.cache.set(request.request_hash, normalized_payload)
        else:
            try:
                self.rate_limiter.wait()
                response = run_with_retries(
                    lambda: self.client.generate(
                        request.prompt,
                        request.system_prompt,
                        self.config.temperature,
                        self.config.max_output_tokens,
                    ),
                    config=RetryConfig(max_retries=self.config.max_retries),
                )
                raw_payload = parse_json_response(response.text)
                normalized_payload = normalize_teacher_payload(
                    raw_payload,
                    request=request,
                    provider_name=response.provider_name,
                    model_name=response.model_name,
                    response_request_id=response.request_id,
                    response_raw=response.raw,
                )
                validation = validate_response_payload(
                    normalized_payload,
                    source_prompt=request.prompt,
                )
                if validation.ok and validation.record is not None:
                    self.cache.set(request.request_hash, normalized_payload)
            except Exception as exc:
                self.manifest.mark_failed(request.candidate.candidate_id, str(exc))
                return

        if not validation.ok or validation.record is None:
            self.manifest.mark_failed(
                request.candidate.candidate_id,
                "; ".join(validation.errors),
            )
            return
        self.writer.write(validation.record)
        self.manifest.mark_completed(
            request.candidate.candidate_id,
            validation.record.synthetic_id,
        )

    def _normalize_cached_payload(
        self,
        cached: dict[str, Any],
        request: TeacherRequest,
    ) -> dict[str, Any]:
        return normalize_teacher_payload(
            cached,
            request=request,
            provider_name=str(getattr(self.client, "provider_name", self.config.provider)),
            model_name=str(getattr(self.client, "model_name", self.config.model)),
        )


def load_teacher_engine_config(path: str | Path) -> TeacherEngineConfig:
    data = load_yaml(path)
    root = data.get("teacher_generation", data)
    if not isinstance(root, dict):
        raise ValueError("teacher_generation config must be a mapping")
    storage = data.get("storage", {})
    inputs = data.get("inputs", {})
    return TeacherEngineConfig(
        run_id=str(root.get("run_id", "parent_llm_structuring_001")),
        provider=str(root.get("provider", data.get("provider", "mock"))),
        model=str(root.get("model", data.get("model", "mock-teacher"))),
        max_requests_per_run=int(root.get("max_requests_per_run", 2000)),
        max_concurrency=int(root.get("max_concurrency", 4)),
        max_retries=int(root.get("max_retries", 5)),
        timeout_seconds=int(root.get("timeout_seconds", 60)),
        temperature=float(root.get("temperature", 0.2)),
        max_output_tokens=int(root.get("max_output_tokens", 1200)),
        dry_run=bool(root.get("dry_run", False)),
        estimate_only=bool(root.get("estimate_only", False)),
        storage=dict(storage or {}),
        inputs=dict(inputs or {}),
    )


def create_teacher_client(config: TeacherEngineConfig) -> BaseTeacherClient:
    provider = config.provider.lower()
    if provider == "mock":
        return MockTeacherClient(model_name=config.model)
    if provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleTeacherClient(
            model_name=config.model,
            timeout_seconds=config.timeout_seconds,
        )
    if provider in {"nvidia", "nvidia_nim"}:
        return NvidiaNIMTeacherClient(
            model_name=config.model,
            timeout_seconds=config.timeout_seconds,
        )
    if provider == "gemini":
        return GeminiTeacherClient(model_name=config.model, timeout_seconds=config.timeout_seconds)
    raise ValueError(f"unsupported teacher provider: {config.provider}")


def override_config(
    config: TeacherEngineConfig,
    *,
    provider: str | None = None,
    max_requests: int | None = None,
    dry_run: bool | None = None,
) -> TeacherEngineConfig:
    return TeacherEngineConfig(
        run_id=config.run_id,
        provider=provider or config.provider,
        model="mock-teacher" if provider == "mock" else config.model,
        max_requests_per_run=max_requests
        if max_requests is not None
        else config.max_requests_per_run,
        max_concurrency=config.max_concurrency,
        max_retries=config.max_retries,
        timeout_seconds=config.timeout_seconds,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        dry_run=config.dry_run if dry_run is None else dry_run,
        estimate_only=config.estimate_only,
        storage=dict(config.storage),
        inputs=dict(config.inputs),
    )


def summarize_result(result: TeacherEngineResult) -> str:
    return (
        f"planned={result.planned_requests} completed={result.completed} "
        f"failed={result.failed} dry_run={result.dry_run} output_dir={result.output_dir}"
    )
