from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from app.config import PROJECT_ROOT
from app.llm_client import LLMClientError
from app.llm_postprocess import LLMPostprocessError, run_qwen3_postprocess
from app.review_artifacts import ensure_review_artifacts, update_correction


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def output_dir() -> Path:
    tmp_root = PROJECT_ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        path = Path(tmp)
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Сереус Биотех работает",
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "Сереус", "start": 0.0, "end": 0.4, "score": 0.94, "speaker": "SPEAKER_00"},
                    {"word": "Биотех", "start": 0.5, "end": 0.9, "score": 0.97, "speaker": "SPEAKER_00"},
                ],
            }
        ]
        _write_json(path / "segments.json", segments)
        _write_json(path / "words.json", [word for segment in segments for word in segment["words"]])
        _write_json(path / "result_raw.json", {"segments": segments, "raw": True})
        yield path


class FakeLLMClient:
    def __init__(self, content: dict[str, Any]) -> None:
        self.content = content

    def redacted_request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {"model": "qwen3-8b", "hasApiKey": False, "messages": messages}

    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {"rawResponse": {"choices": [{"message": {"content": self.content}}]}, "content": self.content}


class FailingLLMClient:
    def redacted_request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {"model": "qwen3-8b", "hasApiKey": False}

    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        raise LLMClientError("LM Studio недоступен")


def _llm_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "model": "qwen3-8b",
        "corrections": [
            {
                "segmentId": "seg-000001",
                "category": "ASR_ERROR",
                "severity": "medium",
                "confidence": 0.91,
                "originalText": "Сереус",
                "suggestedText": "Сириус",
                "reason": "ASR исказил название компании.",
                "requiresAudioReview": False,
                "canBatchApply": False,
            }
        ],
    }


def test_llm_postprocess_merges_suggestions_and_preserves_raw_artifacts(output_dir: Path) -> None:
    raw_hashes = {name: _sha256(output_dir / name) for name in ("result_raw.json", "segments.json", "words.json")}

    result = run_qwen3_postprocess(output_dir, client=FakeLLMClient(_llm_payload()))

    assert result["addedCount"] == 1
    assert (Path(result["auditDir"]) / "input.json").is_file()
    assert (Path(result["auditDir"]) / "profile_snapshot.json").is_file()
    assert (Path(result["auditDir"]) / "request_redacted_001.json").is_file()
    assert (Path(result["auditDir"]) / "validated_response.json").is_file()

    corrections = _read_json(output_dir / "review" / "correction_suggestions.json")["corrections"]
    llm_correction = [item for item in corrections if item.get("sourceRunId") == result["runId"]][0]
    assert llm_correction["source"] == "llm_asr_correction"
    assert llm_correction["status"] == "pending"
    assert llm_correction["id"].startswith("corr-")
    assert "startTime" in llm_correction
    assert raw_hashes == {name: _sha256(output_dir / name) for name in raw_hashes}


def test_llm_postprocess_deduplicates_repeated_pending_suggestions(output_dir: Path) -> None:
    first = run_qwen3_postprocess(output_dir, client=FakeLLMClient(_llm_payload()))
    second = run_qwen3_postprocess(output_dir, client=FakeLLMClient(_llm_payload()))

    assert first["addedCount"] == 1
    assert second["addedCount"] == 0
    assert second["skippedCount"] == 1


def test_llm_postprocess_does_not_overwrite_existing_accepted_suggestion(output_dir: Path) -> None:
    result = run_qwen3_postprocess(output_dir, client=FakeLLMClient(_llm_payload()))
    correction_id = result["addedCorrections"][0]["id"]
    update_correction(output_dir, correction_id, {"status": "accepted"})

    second = run_qwen3_postprocess(output_dir, client=FakeLLMClient(_llm_payload()))

    corrections = _read_json(output_dir / "review" / "correction_suggestions.json")["corrections"]
    matching = [item for item in corrections if item["originalText"] == "Сереус" and item["suggestedText"] == "Сириус"]
    assert second["addedCount"] == 0
    assert len(matching) == 1
    assert matching[0]["status"] == "accepted"


def test_failed_llm_postprocess_writes_audit_and_preserves_approved_transcript(output_dir: Path) -> None:
    ensure_review_artifacts(output_dir)
    approved_path = output_dir / "review" / "approved_transcript.json"
    approved_hash = _sha256(approved_path)

    with pytest.raises(LLMPostprocessError) as exc_info:
        run_qwen3_postprocess(output_dir, client=FailingLLMClient())

    audit_dir = exc_info.value.audit_dir
    assert (audit_dir / "error.json").is_file()
    assert _read_json(audit_dir / "metadata.json")["status"] == "failed"
    assert _sha256(approved_path) == approved_hash


def test_invalid_llm_payload_fails_before_merge(output_dir: Path) -> None:
    payload = _llm_payload()
    payload["corrections"][0]["id"] = "bad-model-owned-id"

    with pytest.raises(LLMPostprocessError) as exc_info:
        run_qwen3_postprocess(output_dir, client=FakeLLMClient(payload))

    assert (exc_info.value.audit_dir / "error.json").is_file()
    corrections = _read_json(output_dir / "review" / "correction_suggestions.json")["corrections"]
    assert not any(item.get("source") == "llm_asr_correction" for item in corrections)


def test_wrong_segment_id_is_skipped_without_mutating_review(output_dir: Path) -> None:
    payload = _llm_payload()
    payload["corrections"][0]["segmentId"] = "seg-missing"

    result = run_qwen3_postprocess(output_dir, client=FakeLLMClient(payload))

    assert result["addedCount"] == 0
    assert result["skippedCount"] == 1
    assert result["skippedCorrections"][0]["reason"] == "segment_not_found"


def test_noop_llm_suggestion_is_skipped(output_dir: Path) -> None:
    payload = _llm_payload()
    payload["corrections"][0]["originalText"] = "мочевого"
    payload["corrections"][0]["suggestedText"] = "мочевого"

    result = run_qwen3_postprocess(output_dir, client=FakeLLMClient(payload))

    assert result["addedCount"] == 0
    assert result["skippedCorrections"][0]["reason"] == "noop_text_change"


def test_listen_only_llm_suggestion_is_not_a_text_correction(output_dir: Path) -> None:
    payload = _llm_payload()
    payload["corrections"][0]["category"] = "NEEDS_LISTENING"
    payload["corrections"][0]["suggestedText"] = "Сереус"
    payload["corrections"][0]["requiresAudioReview"] = True

    result = run_qwen3_postprocess(output_dir, client=FakeLLMClient(payload))

    assert result["addedCount"] == 0
    assert result["skippedCorrections"][0]["reason"] == "listen_flag_not_correction"
