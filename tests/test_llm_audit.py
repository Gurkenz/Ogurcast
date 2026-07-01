from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from app.config import PROJECT_ROOT
from app.llm_contract import LLMAuditResponse
from app.review_artifacts import ensure_review_artifacts, merge_llm_audit_response


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
                "text": "У пациента мышленьвазивный рак.",
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "У", "start": 0.0, "end": 0.1, "score": 0.99, "speaker": "SPEAKER_00"},
                    {"word": "мышленьвазивный", "start": 0.5, "end": 1.2, "score": 0.42, "speaker": "SPEAKER_00"},
                ],
            },
            {
                "start": 2.0,
                "end": 2.3,
                "text": "Да.",
                "speaker": "SPEAKER_01",
                "words": [
                    {"word": "Да", "start": 2.0, "end": 2.2, "score": 0.93, "speaker": "SPEAKER_01"},
                ],
            },
        ]
        _write_json(path / "segments.json", segments)
        _write_json(path / "words.json", [word for segment in segments for word in segment["words"]])
        _write_json(path / "result_raw.json", {"segments": segments, "raw": True})
        _write_json(path / "metadata.json", {"project": "Ogurcast", "language": "ru", "diarization_status": "done", "alignment_status": "done"})
        yield path


def _audit_payload() -> dict[str, Any]:
    return {
        "version": "1",
        "task": "diarized_asr_audit",
        "chunkId": "chunk_0001",
        "suggestions": [
            {
                "segmentId": "seg-000001",
                "involvedSegmentIds": [],
                "category": "ASR_OBVIOUS",
                "action": "replace_text",
                "severity": "low",
                "risk": "low",
                "confidence": 0.96,
                "canAutoApply": True,
                "requiresAudioReview": False,
                "originalText": "мышленьвазивный",
                "suggestedText": "мышечно-инвазивный",
                "reason": "Фонетическая ASR-ошибка медицинского термина.",
                "evidence": [{"type": "context", "value": "рак"}],
            },
            {
                "segmentId": "seg-000001",
                "involvedSegmentIds": [],
                "category": "ASR_LOW_CONFIDENCE",
                "action": "mark_audio_review",
                "severity": "medium",
                "risk": "medium",
                "confidence": 0.42,
                "canAutoApply": False,
                "requiresAudioReview": True,
                "originalText": "мышленьвазивный",
                "suggestedText": "",
                "reason": "Низкая уверенность слова с медицинским смыслом.",
                "evidence": [{"type": "word_score", "value": "0.42"}],
            },
            {
                "segmentId": "seg-000002",
                "involvedSegmentIds": ["seg-000001"],
                "category": "DIARIZATION_MICROSEGMENT",
                "action": "keep_unchanged",
                "severity": "medium",
                "risk": "medium",
                "confidence": 0.8,
                "canAutoApply": False,
                "requiresAudioReview": True,
                "originalText": "Да.",
                "suggestedText": "",
                "reason": "Короткая реплика требует проверки speaker boundary.",
                "evidence": [{"type": "metadata", "value": "duration<0.5"}],
            },
        ],
        "speakerMapSuggestions": [],
        "glossary": [
            {
                "term": "мышечно-инвазивный рак",
                "type": "medical_term",
                "canonical": "мышечно-инвазивный рак",
                "status": "requires_audio_review",
                "confidence": 0.72,
                "reason": "Медицинский термин важен для глоссария.",
            }
        ],
        "technicalFlags": [
            {
                "flag": "word_layer_low_confidence",
                "severity": "medium",
                "risk": "medium",
                "reason": "Есть низкоуверенные слова в медицинском фрагменте.",
                "recommendedAction": "Проверить по аудио.",
            }
        ],
        "summary": {
            "hasCriticalIssues": False,
            "autoApplyCount": 1,
            "audioReviewCount": 2,
            "diarizationIssueCount": 1,
            "asrIssueCount": 2,
            "entityIssueCount": 1,
        },
    }


def test_audit_contract_accepts_valid_payload_and_rejects_extra_fields() -> None:
    parsed = LLMAuditResponse.model_validate(_audit_payload())

    assert parsed.task == "diarized_asr_audit"
    assert parsed.suggestions[0].originalText == "мышленьвазивный"

    payload = _audit_payload()
    payload["unexpected"] = True
    with pytest.raises(ValueError):
        LLMAuditResponse.model_validate(payload)


def test_audit_projection_writes_review_artifacts_without_mutating_raw(output_dir: Path) -> None:
    raw_hashes = {name: _sha256(output_dir / name) for name in ("result_raw.json", "segments.json", "words.json")}
    ensure_review_artifacts(output_dir)

    result = merge_llm_audit_response(output_dir, "llm-test", _audit_payload())

    assert result["addedCount"] == 5
    corrections = _read_json(output_dir / "review" / "correction_suggestions.json")["corrections"]
    audio_flags = _read_json(output_dir / "review" / "audio_flags.json")["flags"]
    entities = _read_json(output_dir / "review" / "entity_annotations.json")["entities"]
    audit_suggestions = _read_json(output_dir / "review" / "audit_suggestions.json")["suggestions"]
    technical_flags = _read_json(output_dir / "review" / "technical_flags.json")["flags"]

    assert any(item["suggestedText"] == "мышечно-инвазивный" for item in corrections)
    assert any(item["category"] == "ASR_LOW_CONFIDENCE" for item in audio_flags)
    assert any(item["surface"] == "мышечно-инвазивный рак" for item in entities)
    assert any(item["category"] == "DIARIZATION_MICROSEGMENT" for item in audit_suggestions)
    assert any(item["flag"] == "word_layer_low_confidence" for item in technical_flags)
    assert raw_hashes == {name: _sha256(output_dir / name) for name in raw_hashes}


def test_audit_projection_skips_wrong_segment_and_non_substring(output_dir: Path) -> None:
    payload = _audit_payload()
    payload["suggestions"][0]["originalText"] = "нет такого текста"
    payload["suggestions"][1]["segmentId"] = "seg-missing"

    result = merge_llm_audit_response(output_dir, "llm-test", payload)

    corrections = _read_json(output_dir / "review" / "correction_suggestions.json")["corrections"]
    assert not corrections
    assert result["skippedCount"] >= 2


def test_audit_projection_keeps_speaker_map_suggestions(output_dir: Path) -> None:
    payload = _audit_payload()
    payload["suggestions"] = []
    payload["glossary"] = []
    payload["technicalFlags"] = []
    payload["speakerMapSuggestions"] = [
        {
            "speakerLabel": "SPEAKER_00",
            "proposedRole": "ведущий",
            "proposedName": "Иван",
            "confidence": 0.7,
            "reason": "Спикер представился в начале фрагмента.",
            "requiresReview": True,
        }
    ]

    result = merge_llm_audit_response(output_dir, "llm-test", payload)

    audit_suggestions = _read_json(output_dir / "review" / "audit_suggestions.json")["suggestions"]
    assert result["addedAuditSuggestions"]
    assert any(item["category"] == "SPEAKER_MAP_SUGGESTION" for item in audit_suggestions)
