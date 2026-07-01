from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.file_utils import ensure_inside_project
from app.llm_client import LLMClientError, LLMSettings, OpenAICompatibleLLMClient
from app.llm_contract import LLMAuditResponse, LLMCorrectionResponse
from app.llm_profiles import LLMProfile, get_profile, profile_hash, render_messages
from app.llm_runtime import ensure_llm_runtime
from app.review_artifacts import REVIEW_DIR_NAME, ensure_review_artifacts, merge_llm_audit_response, merge_llm_corrections


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
    auto: bool = False


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
    auto: bool = False
    fingerprint: str | None = None
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
    fingerprint: str | None = None,
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
        auto=request.auto,
        fingerprint=fingerprint,
        audit_dir=str(audit_dir),
    )
    with _lock:
        _runs[run_id] = state
    _write_json(
        audit_dir / "metadata.json",
        {
            "runId": run_id,
            "status": "queued",
            "stage": request.stage,
            "createdAt": state.created_at,
            "auto": request.auto,
            "autoAuditFingerprint": fingerprint,
        },
    )

    thread = threading.Thread(
        target=_run_llm_thread,
        kwargs={
            "run_id": run_id,
            "output_dir": output_dir,
            "request": request,
            "fingerprint": fingerprint,
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
    fingerprint: str | None,
    client_factory: Callable[[LLMSettings], LLMClientProtocol] | None,
) -> None:
    with _lock:
        state = _runs[run_id]
        state.status = "running"
        state.started_at = _now()
    _write_json(
        _audit_dir(output_dir, run_id) / "metadata.json",
        {
            "runId": run_id,
            "status": "running",
            "stage": request.stage,
            "createdAt": state.created_at,
            "startedAt": state.started_at,
            "auto": request.auto,
            "autoAuditFingerprint": fingerprint,
        },
    )
    try:
        summary = run_llm_stage(
            output_dir,
            request=request,
            run_id=run_id,
            fingerprint=fingerprint,
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


def auto_audit_enabled() -> bool:
    return os.getenv("OGURCAST_LLM_AUTO_AUDIT", "true").strip().casefold() not in {"0", "false", "no", "off"}


def start_auto_llm_audit(
    *,
    job_id: str,
    output_dir: Path,
    client_factory: Callable[[LLMSettings], LLMClientProtocol] | None = None,
) -> dict[str, Any]:
    if not auto_audit_enabled():
        return {"status": "disabled", "stage": "diarized_asr_audit"}
    output_dir = ensure_inside_project(Path(output_dir))
    ensure_review_artifacts(output_dir)
    fingerprint = _auto_audit_fingerprint(output_dir)
    existing = _find_existing_auto_audit(output_dir, fingerprint)
    if existing is not None:
        return {"status": "deduplicated", "stage": "diarized_asr_audit", "run": existing}
    return start_llm_run(
        job_id=job_id,
        output_dir=output_dir,
        request=LLMRunRequest(stage="diarized_asr_audit", auto=True),
        fingerprint=fingerprint,
        client_factory=client_factory,
    )


def run_llm_stage(
    output_dir: Path,
    *,
    request: LLMRunRequest | None = None,
    run_id: str | None = None,
    fingerprint: str | None = None,
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
                auto=request.auto,
                fingerprint=fingerprint,
            )

    try:
        _event(run_id, "started", stage=request.stage)
        bundle = ensure_review_artifacts(output_dir)
        profile = get_profile(request.stage)
        if request.profileId and request.profileId != profile.profileId:
            raise ValueError("profileId не совпадает с активным профилем стадии.")

        settings = _resolve_settings(profile, request)
        if client is None and client_factory is None:
            runtime_status = ensure_llm_runtime(settings)
            _event(run_id, "runtime_ready", provider=settings.provider, status=runtime_status.get("status"))
        client = client or (client_factory(settings) if client_factory else OpenAICompatibleLLMClient(settings))
        _event(run_id, "model_resolved", model=settings.model, temperature=settings.temperature, timeoutSec=settings.timeout_sec)
        _write_json(audit_dir / "profile_snapshot.json", profile.model_dump(mode="json"))
        _write_json(audit_dir / "run_request.json", request.model_dump(mode="json"))
        _require_loaded_model(client, settings.model)

        chunks = _chunks_for_stage(output_dir, bundle, profile, request.stage)
        _write_json(audit_dir / "input.json", {"version": 1, "stage": request.stage, "chunks": chunks})

        validated_payloads: list[dict[str, Any]] = []
        chunk_hashes = []
        for chunk_index, input_payload in enumerate(chunks, start=1):
            messages = render_messages(profile, input_payload, settings.model)
            prompt_digest = profile_hash(profile, messages)
            chunk_hashes.append(prompt_digest)
            suffix = f"{chunk_index:03d}"
            _write_json(audit_dir / f"messages_{suffix}.json", messages)
            _write_json(audit_dir / f"request_redacted_{suffix}.json", _redacted_request(client, messages))
            _event(run_id, "request_written", chunkIndex=chunk_index, chunkTotal=len(chunks), promptHash=prompt_digest)

            completion = _complete_json_with_retries(client, messages, run_id, chunk_index, retry_count=2)
            _write_json(audit_dir / f"raw_response_{suffix}.json", completion.get("rawResponse", completion))
            _event(run_id, "response_received", chunkIndex=chunk_index)

            validated = _validate_stage_response(request.stage, completion.get("content"))
            validated_payload = validated.model_dump(mode="json")
            _write_json(audit_dir / f"validated_response_{suffix}.json", validated_payload)
            _event(run_id, "validated", chunkIndex=chunk_index, itemCount=_validated_item_count(request.stage, validated_payload))
            validated_payloads.append(validated_payload)

        merge_result = _merge_stage_payloads(output_dir, run_id, request.stage, validated_payloads)
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
            "auto": request.auto,
            "autoAuditFingerprint": fingerprint,
        }
        _write_json(audit_dir / "validated_response.json", {"stage": request.stage, "responses": validated_payloads})
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
            **_public_merge_result(merge_result),
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
                "auto": request.auto,
                "autoAuditFingerprint": fingerprint,
            },
        )
        with _lock:
            state = _runs.get(run_id)
            if state is not None:
                state.status = "failed"
                state.finished_at = _now()
                state.error = message
        raise LLMPostprocessError(message, run_id=run_id, audit_dir=audit_dir) from exc


def _chunks_for_stage(output_dir: Path, bundle: dict[str, Any], profile: LLMProfile, stage: str) -> list[dict[str, Any]]:
    if stage == "diarized_asr_audit":
        return _audit_chunks(output_dir, bundle, profile)
    segments = _segments_from_bundle(bundle)
    chunks = _segment_chunks(segments, profile.maxInputChars)
    return [
        {
            "version": 1,
            "stage": stage,
            "chunkIndex": chunk_index,
            "chunkTotal": len(chunks),
            "segments": chunk,
        }
        for chunk_index, chunk in enumerate(chunks, start=1)
    ]


def _validate_stage_response(stage: str, content: Any) -> BaseModel:
    if stage == "diarized_asr_audit":
        return LLMAuditResponse.model_validate(content)
    return LLMCorrectionResponse.model_validate(content)


def _validated_item_count(stage: str, payload: dict[str, Any]) -> int:
    if stage == "diarized_asr_audit":
        return (
            len(payload.get("suggestions", []))
            + len(payload.get("speakerMapSuggestions", []))
            + len(payload.get("glossary", []))
            + len(payload.get("technicalFlags", []))
        )
    return len(payload.get("corrections", []))


def _merge_stage_payloads(output_dir: Path, run_id: str, stage: str, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if stage == "diarized_asr_audit":
        aggregate = _empty_merge_result()
        for payload in payloads:
            _accumulate_merge_result(
                aggregate,
                merge_llm_audit_response(output_dir, run_id, payload, source=f"llm_{stage}"),
            )
        return aggregate

    suggestions = []
    for payload in payloads:
        suggestions.extend(payload.get("corrections", []))
    return merge_llm_corrections(output_dir, run_id, suggestions, source=f"llm_{stage}")


def _empty_merge_result() -> dict[str, Any]:
    return {
        "addedCount": 0,
        "skippedCount": 0,
        "addedCorrections": [],
        "addedAudioFlags": [],
        "addedEntities": [],
        "addedAuditSuggestions": [],
        "addedTechnicalFlags": [],
        "skippedSuggestions": [],
    }


def _accumulate_merge_result(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["addedCount"] += int(source.get("addedCount") or 0)
    target["skippedCount"] += int(source.get("skippedCount") or 0)
    for key in (
        "addedCorrections",
        "addedAudioFlags",
        "addedEntities",
        "addedAuditSuggestions",
        "addedTechnicalFlags",
        "skippedSuggestions",
    ):
        target.setdefault(key, []).extend(source.get(key, []))


def _public_merge_result(merge_result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in merge_result.items()
        if key.startswith("added") or key.startswith("skipped")
    }


def _complete_json_with_retries(
    client: LLMClientProtocol,
    messages: list[dict[str, str]],
    run_id: str,
    chunk_index: int,
    retry_count: int,
) -> dict[str, Any]:
    last_error: LLMClientError | None = None
    for attempt in range(retry_count + 1):
        try:
            return client.complete_json(messages)
        except LLMClientError as exc:
            last_error = exc
            if attempt >= retry_count:
                break
            delay = 1.0 + attempt * 2.0
            _event(run_id, "retry", chunkIndex=chunk_index, attempt=attempt + 1, delaySec=delay, error=str(exc))
            time.sleep(delay)
    raise last_error or LLMClientError("LLM call failed.")


def _auto_audit_fingerprint(output_dir: Path) -> str:
    profile = get_profile("diarized_asr_audit")
    settings = LLMSettings.from_env()
    payload = {
        "segments": _file_sha256(output_dir / "segments.json"),
        "words": _file_sha256(output_dir / "words.json"),
        "profile": profile_hash(profile, []),
        "model": settings.model,
        "provider": settings.provider,
        "llamaModelPath": settings.llama_model_path,
    }
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _find_existing_auto_audit(output_dir: Path, fingerprint: str) -> dict[str, Any] | None:
    runs_dir = _review_dir(output_dir) / "llm_runs"
    if not runs_dir.exists():
        return None
    for metadata_path in runs_dir.glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            isinstance(metadata, dict)
            and metadata.get("stage") == "diarized_asr_audit"
            and metadata.get("autoAuditFingerprint") == fingerprint
            and metadata.get("status") in {"queued", "running", "done"}
        ):
            metadata.setdefault("auditDir", str(metadata_path.parent))
            return metadata
    return None


def _resolve_settings(profile: LLMProfile, request: LLMRunRequest) -> LLMSettings:
    env_settings = LLMSettings.from_env()
    model = (request.model or profile.defaultModel or env_settings.model).strip()
    return LLMSettings(
        base_url=env_settings.base_url,
        model=model,
        temperature=request.temperature if request.temperature is not None else profile.temperature or env_settings.temperature,
        timeout_sec=request.timeoutSec if request.timeoutSec is not None else profile.timeoutSec or env_settings.timeout_sec,
        api_key=env_settings.api_key,
        provider=env_settings.provider,
        llama_server_exe=env_settings.llama_server_exe,
        llama_model_path=env_settings.llama_model_path,
    )


def _require_loaded_model(client: LLMClientProtocol, model: str) -> None:
    status_reader = getattr(client, "model_status", None)
    if not callable(status_reader):
        return
    status = status_reader()
    if not status.get("reachable"):
        raise LLMClientError(str(status.get("error") or "LLM endpoint недоступен."))
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


def _audit_chunks(output_dir: Path, bundle: dict[str, Any], profile: LLMProfile) -> list[dict[str, Any]]:
    segments = _audit_segments_from_bundle(bundle)
    metadata = _metadata_from_output(output_dir)
    technical_signals = _technical_signals(metadata, segments, bundle)
    speaker_turn_text = [
        {
            "speaker": turn.get("speaker"),
            "text": turn.get("text"),
        }
        for turn in bundle.get("speakerTurns", {}).get("turns", [])
        if isinstance(turn, dict)
    ]
    known_speakers = [
        {
            "speakerLabel": speaker.get("label"),
            "role": "unknown",
            "name": speaker.get("displayName") or "",
            "confidence": 0.0 if not speaker.get("displayName") else 0.8,
        }
        for speaker in bundle.get("speakers", {}).get("speakers", [])
        if isinstance(speaker, dict)
    ]
    chunk_size = profile.chunkSegmentCount
    overlap = min(profile.chunkOverlap, max(0, chunk_size - 1))
    chunks: list[dict[str, Any]] = []
    total = max(1, (len(segments) + chunk_size - 1) // chunk_size)
    for chunk_index, primary_start in enumerate(range(0, len(segments), chunk_size), start=1):
        primary_end = min(len(segments), primary_start + chunk_size)
        chunk_start = max(0, primary_start - overlap)
        chunk_end = min(len(segments), primary_end + overlap)
        chunk_segments = segments[chunk_start:chunk_end]
        low_confidence_words = [
            word
            for segment in chunk_segments
            for word in segment.get("words", [])
            if isinstance(word, dict) and isinstance(word.get("score"), (int, float)) and word["score"] < 0.75
        ]
        chunks.append(
            {
                "task": "diarized_asr_audit",
                "language": metadata.get("language") or "ru",
                "chunkId": f"chunk_{chunk_index:04d}",
                "chunkIndex": chunk_index,
                "chunkTotal": total,
                "chunkRange": {
                    "start": chunk_segments[0].get("start") if chunk_segments else 0.0,
                    "end": chunk_segments[-1].get("end") if chunk_segments else 0.0,
                },
                "primarySegmentIds": [segment["segmentId"] for segment in segments[primary_start:primary_end]],
                "metadata": {
                    "project": metadata.get("project") or "Ogurcast",
                    "model": metadata.get("model"),
                    "language": metadata.get("language"),
                    "alignmentStatus": metadata.get("alignment_status"),
                    "diarizationStatus": metadata.get("diarization_status"),
                    "minSpeakers": metadata.get("min_speakers"),
                    "maxSpeakers": metadata.get("max_speakers"),
                },
                "knownSpeakers": known_speakers,
                "segments": chunk_segments,
                "adjacentContext": {
                    "previousSegments": segments[max(0, chunk_start - overlap):chunk_start],
                    "nextSegments": segments[chunk_end:min(len(segments), chunk_end + overlap)],
                },
                "speakerTurnText": speaker_turn_text,
                "lowConfidenceWords": low_confidence_words,
                "sourceConflicts": [],
                "entityEvidence": [],
                "technicalSignals": technical_signals,
            }
        )
    return chunks or [
        {
            "task": "diarized_asr_audit",
            "language": metadata.get("language") or "ru",
            "chunkId": "chunk_0001",
            "chunkIndex": 1,
            "chunkTotal": 1,
            "chunkRange": {"start": 0.0, "end": 0.0},
            "primarySegmentIds": [],
            "metadata": metadata,
            "knownSpeakers": known_speakers,
            "segments": [],
            "adjacentContext": {"previousSegments": [], "nextSegments": []},
            "speakerTurnText": speaker_turn_text,
            "lowConfidenceWords": [],
            "sourceConflicts": [],
            "entityEvidence": [],
            "technicalSignals": technical_signals,
        }
    ]


def _audit_segments_from_bundle(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = [
        segment
        for segment in bundle.get("transcript", {}).get("segments", [])
        if isinstance(segment, dict)
    ]
    audit_segments = []
    for segment in raw_segments:
        segment_id = str(segment.get("id"))
        audit_segments.append(
            {
                "segmentId": segment_id,
                "start": segment.get("start"),
                "end": segment.get("end"),
                "speaker": segment.get("speaker"),
                "text": segment.get("text"),
                "words": [
                    {
                        "wordId": word.get("id"),
                        "word": word.get("word"),
                        "start": word.get("start"),
                        "end": word.get("end"),
                        "score": word.get("score"),
                        "speaker": word.get("speaker"),
                        "segmentId": segment_id,
                    }
                    for word in segment.get("words", [])
                    if isinstance(word, dict)
                ],
            }
        )
    return audit_segments


def _metadata_from_output(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "metadata.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _technical_signals(metadata: dict[str, Any], segments: list[dict[str, Any]], bundle: dict[str, Any]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    if metadata.get("alignment_status") != "done":
        signals.append({"type": "alignment_not_done", "value": _text_or_none(metadata.get("alignment_status"))})
    if metadata.get("diarization_status") != "done":
        signals.append({"type": "diarization_not_done", "value": _text_or_none(metadata.get("diarization_status"))})
    if not bundle.get("words"):
        signals.append({"type": "missing_word_layer", "value": "words.json is empty"})
    for segment in segments:
        word_speakers = {
            word.get("speaker")
            for word in segment.get("words", [])
            if isinstance(word, dict) and word.get("speaker")
        }
        if len(word_speakers) > 1:
            signals.append(
                {
                    "type": "intrasegment_speaker_shift",
                    "segmentId": segment.get("segmentId"),
                    "value": ", ".join(sorted(str(item) for item in word_speakers)),
                }
            )
        duration = _number(segment.get("end")) - _number(segment.get("start"))
        if 0 < duration < 0.5:
            signals.append(
                {
                    "type": "microsegment",
                    "segmentId": segment.get("segmentId"),
                    "value": f"{duration:.3f}",
                }
            )
    return signals


def _text_or_none(value: Any) -> str | None:
    text = str(value) if value is not None else None
    return text if text else None


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
