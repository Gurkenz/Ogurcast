from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.file_utils import ensure_inside_project
from app.llm_client import LLMClientError, LLMSettings, OpenAICompatibleLLMClient
from app.llm_contract import LLMCorrectionResponse
from app.llm_profiles import LLMProfile, get_profile, profile_hash, render_messages
from app.review_artifacts import REVIEW_DIR_NAME, ensure_review_artifacts, merge_llm_corrections


class LLMPostprocessError(RuntimeError):
    def __init__(self, message: str, *, run_id: str, audit_dir: Path) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.audit_dir = audit_dir


class LLMClientProtocol(Protocol):
    settings: LLMSettings

    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        ...


class LLMRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str = Field(default="asr_correction", min_length=1)
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    timeoutSec: float | None = Field(default=None, gt=0.0, le=600.0)
    profileId: str | None = None


@dataclass
class LLMRunState:
    run_id: str
    job_id: str
    stage: str
    status: str
    output_dir: str
    audit_dir: str
    created_at: str = field(default_factory=lambda: _now())
    started_at: str | None = None
    finished_at: str | None = None
    model: str | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


_lock = threading.RLock()
_runs: dict[str, LLMRunState] = {}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _review_dir(output_dir: Path) -> Path:
    return ensure_inside_project(output_dir) / REVIEW_DIR_NAME


def _audit_dir(output_dir: Path, run_id: str) -> Path:
    return _review_dir(output_dir) / "llm_runs" / run_id


def _event(run_id: str, event_type: str, **payload: Any) -> dict[str, Any]:
    event = {"time": _now(), "type": event_type, **payload}
    with _lock:
        state = _runs.get(run_id)
        if state is not None:
            state.events.append(event)
            _write_json(Path(state.audit_dir) / "events.json", state.events)
    return event


def _state_dict(state: LLMRunState) -> dict[str, Any]:
    return asdict(state)


def get_llm_run(run_id: str) -> dict[str, Any] | None:
    with _lock:
        state = _runs.get(run_id)
        return _state_dict(state) if state is not None else None


def start_llm_run(
    *,
    job_id: str,
    output_dir: Path,
    request: LLMRunRequest,
    client_factory: Callable[[LLMSettings], LLMClientProtocol] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_inside_project(Path(output_dir))
    run_id = f"llm-{uuid.uuid4().hex[:12]}"
    audit_dir = _audit_dir(output_dir, run_id)
    state = LLMRunState(
        run_id=run_id,
        job_id=job_id,
        stage=request.stage,
        status="queued",
        output_dir=str(output_dir),
        audit_dir=str(audit_dir),
    )
    with _lock:
        _runs[run_id] = state

    thread = threading.Thread(
        target=_run_llm_thread,
        kwargs={
            "run_id": run_id,
            "output_dir": output_dir,
            "request": request,
            "client_factory": client_factory,
        },
        daemon=True,
    )
    thread.start()
    return _state_dict(state)


def _run_llm_thread(
    *,
    run_id: str,
    output_dir: Path,
    request: LLMRunRequest,
    client_factory: Callable[[LLMSettings], LLMClientProtocol] | None,
) -> None:
    with _lock:
        state = _runs[run_id]
        state.status = "running"
        state.started_at = _now()
    try:
        summary = run_llm_stage(
            output_dir,
            request=request,
            run_id=run_id,
            client_factory=client_factory,
        )
        with _lock:
            state = _runs[run_id]
            state.status = "done"
            state.finished_at = _now()
            state.model = summary.get("model")
            state.summary = summary
    except LLMPostprocessError as exc:
        with _lock:
            state = _runs[run_id]
            state.status = "failed"
            state.finished_at = _now()
            state.error = str(exc)
    except Exception as exc:
        message = str(exc)
        _event(run_id, "failed", error=message)
        audit_dir = _audit_dir(output_dir, run_id)
        _write_json(audit_dir / "error.json", {"runId": run_id, "errorType": exc.__class__.__name__, "message": message})
        with _lock:
            state = _runs[run_id]
            state.status = "failed"
            state.finished_at = _now()
            state.error = message


def run_llm_stage(
    output_dir: Path,
    *,
    request: LLMRunRequest | None = None,
    run_id: str | None = None,
    client: LLMClientProtocol | None = None,
    client_factory: Callable[[LLMSettings], LLMClientProtocol] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_inside_project(Path(output_dir))
    request = request or LLMRunRequest()
    run_id = run_id or f"llm-{uuid.uuid4().hex[:12]}"
    audit_dir = _audit_dir(output_dir, run_id)
    started_at = _now()

    with _lock:
        if run_id not in _runs:
            _runs[run_id] = LLMRunState(
                run_id=run_id,
                job_id="sync",
                stage=request.stage,
                status="running",
                output_dir=str(output_dir),
                audit_dir=str(audit_dir),
                started_at=started_at,
            )

    try:
        _event(run_id, "started", stage=request.stage)
        bundle = ensure_review_artifacts(output_dir)
        profile = get_profile(request.stage)
        if request.profileId and request.profileId != profile.profileId:
            raise ValueError("profileId не совпадает с активным профилем стадии.")

        settings = _resolve_settings(profile, request)
        client = client or (client_factory(settings) if client_factory else OpenAICompatibleLLMClient(settings))
        _event(run_id, "model_resolved", model=settings.model, temperature=settings.temperature, timeoutSec=settings.timeout_sec)
        _write_json(audit_dir / "profile_snapshot.json", profile.model_dump(mode="json"))
        _write_json(audit_dir / "run_request.json", request.model_dump(mode="json"))
        _require_loaded_model(client, settings.model)

        segments = _segments_from_bundle(bundle)
        chunks = _segment_chunks(segments, profile.maxInputChars)
        _write_json(audit_dir / "input.json", {"version": 1, "chunks": chunks})

        validated_suggestions: list[dict[str, Any]] = []
        chunk_hashes = []
        for chunk_index, chunk_segments in enumerate(chunks, start=1):
            input_payload = {
                "version": 1,
                "stage": request.stage,
                "chunkIndex": chunk_index,
                "chunkTotal": len(chunks),
                "segments": chunk_segments,
            }
            messages = render_messages(profile, input_payload, settings.model)
            prompt_digest = profile_hash(profile, messages)
            chunk_hashes.append(prompt_digest)
            suffix = f"{chunk_index:03d}"
            _write_json(audit_dir / f"messages_{suffix}.json", messages)
            _write_json(audit_dir / f"request_redacted_{suffix}.json", _redacted_request(client, messages))
            _event(run_id, "request_written", chunkIndex=chunk_index, chunkTotal=len(chunks), promptHash=prompt_digest)

            completion = client.complete_json(messages)
            _write_json(audit_dir / f"raw_response_{suffix}.json", completion.get("rawResponse", completion))
            _event(run_id, "response_received", chunkIndex=chunk_index)

            validated = LLMCorrectionResponse.model_validate(completion.get("content"))
            validated_payload = validated.model_dump(mode="json")
            _write_json(audit_dir / f"validated_response_{suffix}.json", validated_payload)
            _event(run_id, "validated", chunkIndex=chunk_index, correctionCount=len(validated.corrections))
            validated_suggestions.extend(correction.model_dump(mode="json") for correction in validated.corrections)

        merge_result = merge_llm_corrections(
            output_dir,
            run_id,
            validated_suggestions,
            source=f"llm_{request.stage}",
        )
        _event(
            run_id,
            "merged",
            addedCount=merge_result["addedCount"],
            skippedCount=merge_result["skippedCount"],
        )
        metadata = {
            "runId": run_id,
            "status": "done",
            "stage": request.stage,
            "startedAt": started_at,
            "finishedAt": _now(),
            "model": settings.model,
            "temperature": settings.temperature,
            "timeoutSec": settings.timeout_sec,
            "profileId": profile.profileId,
            "profileVersion": profile.version,
            "promptHashes": chunk_hashes,
            "chunkCount": len(chunks),
            "addedCount": merge_result["addedCount"],
            "skippedCount": merge_result["skippedCount"],
        }
        _write_json(audit_dir / "validated_response.json", {"corrections": validated_suggestions})
        _write_json(audit_dir / "metadata.json", metadata)
        with _lock:
            state = _runs.get(run_id)
            if state is not None:
                state.status = "done"
                state.finished_at = metadata["finishedAt"]
                state.model = settings.model
                state.summary = metadata
        return {
            **metadata,
            "auditDir": str(audit_dir),
            "addedCorrections": merge_result["addedCorrections"],
            "skippedCorrections": merge_result["skippedCorrections"],
        }
    except (LLMClientError, ValidationError, ValueError, KeyError) as exc:
        message = _error_message(exc)
        _event(run_id, "failed", error=message)
        _write_json(
            audit_dir / "error.json",
            {
                "runId": run_id,
                "errorType": exc.__class__.__name__,
                "message": message,
            },
        )
        _write_json(
            audit_dir / "metadata.json",
            {
                "runId": run_id,
                "status": "failed",
                "stage": request.stage,
                "startedAt": started_at,
                "finishedAt": _now(),
                "error": message,
            },
        )
        with _lock:
            state = _runs.get(run_id)
            if state is not None:
                state.status = "failed"
                state.finished_at = _now()
                state.error = message
        raise LLMPostprocessError(message, run_id=run_id, audit_dir=audit_dir) from exc


def _resolve_settings(profile: LLMProfile, request: LLMRunRequest) -> LLMSettings:
    env_settings = LLMSettings.from_env()
    model = (request.model or profile.defaultModel or env_settings.model).strip()
    return LLMSettings(
        base_url=env_settings.base_url,
        model=model,
        temperature=request.temperature if request.temperature is not None else profile.temperature or env_settings.temperature,
        timeout_sec=request.timeoutSec if request.timeoutSec is not None else profile.timeoutSec or env_settings.timeout_sec,
        api_key=env_settings.api_key,
    )


def _require_loaded_model(client: LLMClientProtocol, model: str) -> None:
    status_reader = getattr(client, "model_status", None)
    if not callable(status_reader):
        return
    status = status_reader()
    if not status.get("reachable"):
        raise LLMClientError(str(status.get("error") or "LM Studio недоступен."))
    if not status.get("modelLoaded"):
        available = ", ".join(status.get("availableModels") or [])
        suffix = f" Доступно: {available}." if available else ""
        raise LLMClientError(f"Модель LLM не загружена или имя не совпадает: {model}.{suffix}")


def _redacted_request(client: LLMClientProtocol, messages: list[dict[str, str]]) -> dict[str, Any]:
    redactor = getattr(client, "redacted_request", None)
    if callable(redactor):
        return redactor(messages)
    settings = getattr(client, "settings", None)
    return {
        "client": client.__class__.__name__,
        "model": getattr(settings, "model", None),
        "messages": messages,
    }


def _segments_from_bundle(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = [
        segment
        for segment in bundle.get("transcript", {}).get("segments", [])
        if isinstance(segment, dict)
    ]
    turn_by_segment: dict[str, dict[str, Any]] = {}
    for turn in bundle.get("speakerTurns", {}).get("turns", []):
        if not isinstance(turn, dict):
            continue
        for segment_id in turn.get("segmentIds", []):
            turn_by_segment[str(segment_id)] = turn

    segments = []
    for index, segment in enumerate(raw_segments):
        segment_id = str(segment.get("id"))
        previous_segment = raw_segments[index - 1] if index > 0 else None
        next_segment = raw_segments[index + 1] if index + 1 < len(raw_segments) else None
        low_confidence_words = [
            {
                "word": word.get("word"),
                "start": word.get("start"),
                "end": word.get("end"),
                "score": word.get("score"),
            }
            for word in segment.get("words", [])
            if isinstance(word, dict) and isinstance(word.get("score"), (int, float)) and word.get("score") < 0.75
        ]
        turn = turn_by_segment.get(segment_id)
        segments.append(
            {
                "id": segment.get("id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "speaker": segment.get("speaker"),
                "text": segment.get("text"),
                "lowConfidenceWords": low_confidence_words,
                "adjacentContext": {
                    "previous": _context_segment(previous_segment),
                    "next": _context_segment(next_segment),
                },
                "speakerTurnText": turn.get("text") if isinstance(turn, dict) else None,
            }
        )
    return segments


def _context_segment(segment: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(segment, dict):
        return None
    return {
        "id": segment.get("id"),
        "speaker": segment.get("speaker"),
        "text": segment.get("text"),
    }


def _segment_chunks(segments: list[dict[str, Any]], max_chars: int) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 0
    for segment in segments:
        size = len(json.dumps(segment, ensure_ascii=False))
        if current and current_size + size > max_chars:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(segment)
        current_size += size
    if current:
        chunks.append(current)
    return chunks or [[]]


def _error_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "LLM response не прошел строгую JSON-валидацию."
    return str(exc)
