from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from app.file_utils import ensure_inside_project
from app.llm_client import LLMClientError, OpenAICompatibleLLMClient
from app.llm_contract import LLMCorrectionResponse
from app.review_artifacts import REVIEW_DIR_NAME, ensure_review_artifacts, merge_llm_corrections


class LLMPostprocessError(RuntimeError):
    def __init__(self, message: str, *, run_id: str, audit_dir: Path) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.audit_dir = audit_dir


class LLMClientProtocol(Protocol):
    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        ...


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _review_dir(output_dir: Path) -> Path:
    return ensure_inside_project(output_dir) / REVIEW_DIR_NAME


def _audit_dir(output_dir: Path, run_id: str) -> Path:
    return _review_dir(output_dir) / "llm_runs" / run_id


def _prompt_messages(input_payload: dict[str, Any]) -> list[dict[str, str]]:
    schema = {
        "version": 1,
        "model": "qwen3-8b",
        "corrections": [
            {
                "segmentId": "seg-000001",
                "category": "ASR_ERROR|TYPO|PUNCTUATION|FILLER_GARBAGE|NEEDS_LISTENING",
                "severity": "low|medium|high",
                "confidence": 0.0,
                "originalText": "text from the segment",
                "suggestedText": "replacement text",
                "reason": "short Russian explanation",
                "requiresAudioReview": False,
                "canBatchApply": False,
            }
        ],
    }
    system = (
        "Ты внутренний редактор русскоязычной ASR-расшифровки Ogurcast. "
        "Верни только валидный JSON, без markdown и пояснений вне JSON. "
        "Не переписывай сегменты целиком. Не меняй speaker, start/end и segmentId. "
        "Предлагай только точечные CorrectionSuggestion для очевидных ASR-ошибок, опечаток, пунктуации, мусора распознавания "
        "или мест, требующих прослушивания. Если уверенности нет, ставь requiresAudioReview=true и canBatchApply=false. "
        "Запрещено добавлять поля id, status, source, sourceRunId, startTime, endTime."
    )
    user = json.dumps(
        {
            "task": "return_correction_suggestions",
            "output_schema": schema,
            "input": input_payload,
        },
        ensure_ascii=False,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _redacted_request(client: LLMClientProtocol, messages: list[dict[str, str]]) -> dict[str, Any]:
    redactor = getattr(client, "redacted_request", None)
    if callable(redactor):
        return redactor(messages)
    return {"client": client.__class__.__name__, "messages": messages}


def run_qwen3_postprocess(output_dir: Path, client: LLMClientProtocol | None = None) -> dict[str, Any]:
    output_dir = ensure_inside_project(Path(output_dir))
    bundle = ensure_review_artifacts(output_dir)
    run_id = f"llm-{uuid.uuid4().hex[:12]}"
    audit_dir = _audit_dir(output_dir, run_id)
    started_at = _now()
    client = client or OpenAICompatibleLLMClient()

    segments = [
        {
            "id": segment.get("id"),
            "start": segment.get("start"),
            "end": segment.get("end"),
            "speaker": segment.get("speaker"),
            "text": segment.get("text"),
        }
        for segment in bundle.get("transcript", {}).get("segments", [])
        if isinstance(segment, dict)
    ]
    input_payload = {"version": 1, "segments": segments}
    messages = _prompt_messages(input_payload)

    try:
        _write_json(audit_dir / "input.json", input_payload)
        _write_json(audit_dir / "messages.json", messages)
        _write_json(audit_dir / "request_redacted.json", _redacted_request(client, messages))

        completion = client.complete_json(messages)
        _write_json(audit_dir / "raw_response.json", completion.get("rawResponse", completion))

        validated = LLMCorrectionResponse.model_validate(completion.get("content"))
        validated_payload = validated.model_dump(mode="json")
        _write_json(audit_dir / "validated_response.json", validated_payload)

        merge_result = merge_llm_corrections(
            output_dir,
            run_id,
            [correction.model_dump(mode="json") for correction in validated.corrections],
        )
        metadata = {
            "runId": run_id,
            "status": "done",
            "startedAt": started_at,
            "finishedAt": _now(),
            "model": validated.model,
            "addedCount": merge_result["addedCount"],
            "skippedCount": merge_result["skippedCount"],
        }
        _write_json(audit_dir / "metadata.json", metadata)
        return {
            **metadata,
            "auditDir": str(audit_dir),
            "addedCorrections": merge_result["addedCorrections"],
            "skippedCorrections": merge_result["skippedCorrections"],
        }
    except (LLMClientError, ValidationError, ValueError, KeyError) as exc:
        message = _error_message(exc)
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
                "startedAt": started_at,
                "finishedAt": _now(),
                "error": message,
            },
        )
        raise LLMPostprocessError(message, run_id=run_id, audit_dir=audit_dir) from exc


def _error_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "Qwen3 response не прошел строгую JSON-валидацию."
    return str(exc)
