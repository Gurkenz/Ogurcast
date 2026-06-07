from __future__ import annotations

import json
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from app.file_utils import ensure_inside_project


REVIEW_DIR_NAME = "review"
SAFE_BATCH_CATEGORIES = {"ASR_ERROR", "TYPO", "PUNCTUATION"}
CORRECTION_STATUSES = {"pending", "accepted", "rejected", "modified"}
LLM_SOURCE = "llm_qwen3"
ENTITY_STATUSES = {"pending", "accepted", "rejected", "modified"}
ENTITY_VERIFICATION_STATUSES = {
    "new",
    "confirmed",
    "uncertain",
    "contradicted",
    "not_found",
    "manual_confirmed",
    "manual_rejected",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _review_dir(output_dir: Path) -> Path:
    output_dir = ensure_inside_project(Path(output_dir))
    return output_dir / REVIEW_DIR_NAME


def _required_output_path(output_dir: Path, name: str) -> Path:
    path = ensure_inside_project(Path(output_dir)) / name
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    return str(value or "")


def _segment_speaker(segment: dict[str, Any]) -> str:
    if segment.get("speaker"):
        return str(segment["speaker"])
    speakers = [
        str(word["speaker"])
        for word in segment.get("words", [])
        if isinstance(word, dict) and word.get("speaker")
    ]
    if speakers:
        return Counter(speakers).most_common(1)[0][0]
    return "UNKNOWN"


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _correction_key(correction: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _text(correction.get("segmentId")),
        _text(correction.get("category")),
        _normalize_whitespace(_text(correction.get("originalText"))).casefold(),
        _normalize_whitespace(_text(correction.get("suggestedText"))).casefold(),
    )


def _next_correction_index(corrections: list[dict[str, Any]]) -> int:
    max_index = 0
    for correction in corrections:
        raw_id = _text(correction.get("id"))
        match = re.fullmatch(r"corr-(\d+)", raw_id)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _word_id(segment_index: int, word_index: int) -> str:
    return f"word-{segment_index + 1:06d}-{word_index + 1:04d}"


def _segment_id(segment_index: int) -> str:
    return f"seg-{segment_index + 1:06d}"


def _normalize_segments(raw_segments: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for segment_index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            continue

        words = []
        for word_index, raw_word in enumerate(raw_segment.get("words", [])):
            if not isinstance(raw_word, dict):
                continue
            words.append(
                {
                    "id": _word_id(segment_index, word_index),
                    "start": _float(raw_word.get("start")),
                    "end": _float(raw_word.get("end")),
                    "word": _text(raw_word.get("word")).strip(),
                    "speaker": raw_word.get("speaker") or raw_segment.get("speaker") or "UNKNOWN",
                    "score": raw_word.get("score"),
                }
            )

        normalized.append(
            {
                "id": _segment_id(segment_index),
                "sourceIndex": segment_index,
                "start": _float(raw_segment.get("start")),
                "end": _float(raw_segment.get("end")),
                "speaker": _segment_speaker(raw_segment),
                "text": _text(raw_segment.get("text")),
                "words": words,
            }
        )
    return normalized


def _new_correction(
    *,
    index: int,
    segment_id: str,
    category: str,
    severity: str,
    confidence: float,
    original_text: str,
    suggested_text: str,
    reason: str,
    requires_audio_review: bool,
    can_batch_apply: bool,
    start_time: float | None = None,
    end_time: float | None = None,
    word_start_id: str | None = None,
    word_end_id: str | None = None,
) -> dict[str, Any]:
    correction: dict[str, Any] = {
        "id": f"corr-{index:06d}",
        "segmentId": segment_id,
        "category": category,
        "severity": severity,
        "confidence": round(confidence, 3),
        "originalText": original_text,
        "suggestedText": suggested_text,
        "reason": reason,
        "requiresAudioReview": requires_audio_review,
        "canBatchApply": can_batch_apply,
        "status": "pending",
    }
    if start_time is not None:
        correction["startTime"] = start_time
    if end_time is not None:
        correction["endTime"] = end_time
    if word_start_id is not None:
        correction["wordStartId"] = word_start_id
    if word_end_id is not None:
        correction["wordEndId"] = word_end_id
    return correction


def _generate_corrections(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corrections = []
    next_index = 1

    for segment in segments:
        original = segment["text"]
        normalized = _normalize_whitespace(original)
        if original and normalized and normalized != original:
            corrections.append(
                _new_correction(
                    index=next_index,
                    segment_id=segment["id"],
                    category="PUNCTUATION",
                    severity="low",
                    confidence=0.99,
                    original_text=original,
                    suggested_text=normalized,
                    reason="Безопасная нормализация пробелов внутри сегмента.",
                    requires_audio_review=False,
                    can_batch_apply=True,
                    start_time=segment["start"],
                    end_time=segment["end"],
                )
            )
            next_index += 1

        for word in segment["words"]:
            score = word.get("score")
            if not isinstance(score, (int, float)) or score >= 0.75:
                continue
            word_text = word.get("word") or ""
            if not word_text:
                continue
            corrections.append(
                _new_correction(
                    index=next_index,
                    segment_id=segment["id"],
                    category="NEEDS_LISTENING",
                    severity="medium",
                    confidence=max(0.0, min(1.0, float(score))),
                    original_text=word_text,
                    suggested_text=word_text,
                    reason="Низкая уверенность ASR; требуется прослушивание.",
                    requires_audio_review=True,
                    can_batch_apply=False,
                    start_time=word["start"],
                    end_time=word["end"],
                    word_start_id=word["id"],
                    word_end_id=word["id"],
                )
            )
            next_index += 1

    return corrections


def _entity_type(surface: str) -> str:
    lowered = surface.lower()
    if re.search(r"\d", surface):
        return "DATE" if re.search(r"\b20\d{2}\b", surface) else "NUMBER"
    if any(marker in lowered for marker in ("ооо", "ао", "нпо", "био", "медиа", "групп", "компания", "google")):
        return "ORG"
    if len(surface.split()) == 2:
        return "PERSON"
    return "TERM"


def _entity_key(surface: str) -> str:
    return re.sub(r"\s+", " ", surface).strip(" «»\"'.,:;!?").casefold()


def _entity_candidates(text: str) -> list[str]:
    candidates = []
    candidates.extend(match.group(1) for match in re.finditer(r"«([^»]{3,80})»", text))
    candidates.extend(
        match.group(0)
        for match in re.finditer(
            r"\b[А-ЯЁ][а-яё0-9-]+(?:\s+[А-ЯЁ][а-яё0-9-]+){1,3}\b",
            text,
        )
    )
    return candidates


def _generate_entities(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    segment_ids_by_key: dict[str, set[str]] = defaultdict(set)

    for segment in segments:
        for raw_surface in _entity_candidates(segment["text"]):
            surface = re.sub(r"\s+", " ", raw_surface).strip()
            if len(surface) < 3:
                continue
            key = _entity_key(surface)
            if not key:
                continue
            if key not in found:
                found[key] = {
                    "surface": surface,
                    "canonical": surface,
                    "type": _entity_type(surface),
                    "verificationStatus": "uncertain",
                    "verifierConfidence": 0.62,
                    "evidence": [
                        {
                            "source": "local_transcript",
                            "snippet": _normalize_whitespace(segment["text"])[:180],
                        }
                    ],
                    "status": "pending",
                }
            segment_ids_by_key[key].add(segment["id"])

    entities = []
    for index, key in enumerate(sorted(found), start=1):
        entity = found[key]
        entity["id"] = f"ent-{index:06d}"
        entity["segmentIds"] = sorted(segment_ids_by_key[key])
        entities.append(entity)
    return entities


def _generate_speakers(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    speakers = sorted({segment["speaker"] for segment in segments if segment.get("speaker")})
    return [
        {
            "id": f"speaker-{index:03d}",
            "label": speaker,
            "displayName": None,
            "linkedEntityId": None,
            "verificationStatus": "new",
        }
        for index, speaker in enumerate(speakers, start=1)
    ]


def _approved_transcript(segments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": 1,
        "updatedAt": _now(),
        "segments": [
            {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "speaker": segment["speaker"],
                "text": segment["text"],
            }
            for segment in segments
        ],
    }


def ensure_review_artifacts(output_dir: Path) -> dict[str, Any]:
    output_dir = ensure_inside_project(Path(output_dir))
    raw_segments = _read_json(_required_output_path(output_dir, "segments.json"))
    _required_output_path(output_dir, "words.json")
    review_dir = _review_dir(output_dir)
    created_at = _now()

    transcript_path = review_dir / "transcript_normalized.json"
    corrections_path = review_dir / "correction_suggestions.json"
    entities_path = review_dir / "entity_annotations.json"
    speakers_path = review_dir / "speaker_profiles.json"
    batches_path = review_dir / "edit_batches.json"
    approved_path = review_dir / "approved_transcript.json"
    state_path = review_dir / "review_state.json"

    if not transcript_path.exists():
        segments = _normalize_segments(raw_segments if isinstance(raw_segments, list) else [])
        _write_json(transcript_path, {"version": 1, "createdAt": created_at, "segments": segments})
    else:
        segments = _read_json(transcript_path).get("segments", [])

    if not corrections_path.exists():
        _write_json(corrections_path, {"version": 1, "updatedAt": created_at, "corrections": _generate_corrections(segments)})
    if not entities_path.exists():
        _write_json(entities_path, {"version": 1, "updatedAt": created_at, "entities": _generate_entities(segments)})
    if not speakers_path.exists():
        _write_json(speakers_path, {"version": 1, "updatedAt": created_at, "speakers": _generate_speakers(segments)})
    if not batches_path.exists():
        _write_json(batches_path, {"version": 1, "updatedAt": created_at, "batches": []})
    if not approved_path.exists():
        _write_json(approved_path, _approved_transcript(segments))
    if not state_path.exists():
        _write_json(
            state_path,
            {
                "version": 1,
                "createdAt": created_at,
                "updatedAt": created_at,
                "activeMode": "text",
                "artifactSource": "deterministic_mock",
            },
        )

    return get_review_bundle(output_dir)


def review_artifact_paths(output_dir: Path) -> dict[str, str]:
    review_dir = _review_dir(output_dir)
    names = (
        "transcript_normalized.json",
        "correction_suggestions.json",
        "entity_annotations.json",
        "speaker_profiles.json",
        "edit_batches.json",
        "approved_transcript.json",
        "review_state.json",
    )
    paths = {name: str(review_dir / name) for name in names}
    paths["llm_runs"] = str(review_dir / "llm_runs")
    return paths


def get_review_bundle(output_dir: Path) -> dict[str, Any]:
    review_dir = _review_dir(output_dir)
    return {
        "outputDir": str(ensure_inside_project(Path(output_dir))),
        "reviewDir": str(review_dir),
        "transcript": _read_json(review_dir / "transcript_normalized.json"),
        "approvedTranscript": _read_json(review_dir / "approved_transcript.json"),
        "corrections": _read_json(review_dir / "correction_suggestions.json"),
        "entities": _read_json(review_dir / "entity_annotations.json"),
        "speakers": _read_json(review_dir / "speaker_profiles.json"),
        "editBatches": _read_json(review_dir / "edit_batches.json"),
        "reviewState": _read_json(review_dir / "review_state.json"),
    }


def get_transcript(output_dir: Path) -> dict[str, Any]:
    ensure_review_artifacts(output_dir)
    return _read_json(_review_dir(Path(output_dir)) / "transcript_normalized.json")


def get_words(output_dir: Path) -> list[Any]:
    ensure_review_artifacts(output_dir)
    return _read_json(_required_output_path(Path(output_dir), "words.json"))


def get_entities(output_dir: Path) -> dict[str, Any]:
    ensure_review_artifacts(output_dir)
    return _read_json(_review_dir(Path(output_dir)) / "entity_annotations.json")


def merge_llm_corrections(output_dir: Path, run_id: str, suggestions: list[dict[str, Any]]) -> dict[str, Any]:
    ensure_review_artifacts(output_dir)
    review_dir = _review_dir(Path(output_dir))
    corrections_path = review_dir / "correction_suggestions.json"
    transcript = _read_json(review_dir / "transcript_normalized.json")
    corrections_data = _read_json(corrections_path)
    corrections = corrections_data.setdefault("corrections", [])
    if not isinstance(corrections, list):
        raise ValueError("correction_suggestions.json поврежден.")

    segments = {
        segment["id"]: segment
        for segment in transcript.get("segments", [])
        if isinstance(segment, dict) and segment.get("id")
    }
    existing_keys = {
        _correction_key(correction)
        for correction in corrections
        if isinstance(correction, dict)
    }
    next_index = _next_correction_index([item for item in corrections if isinstance(item, dict)])
    added = []
    skipped = []

    for suggestion in suggestions:
        segment_id = _text(suggestion.get("segmentId"))
        segment = segments.get(segment_id)
        if segment is None:
            skipped.append({"segmentId": segment_id, "reason": "segment_not_found"})
            continue
        key = _correction_key(suggestion)
        if key in existing_keys:
            skipped.append({"segmentId": segment_id, "reason": "duplicate"})
            continue

        correction = {
            "id": f"corr-{next_index:06d}",
            "segmentId": segment_id,
            "category": suggestion["category"],
            "severity": suggestion["severity"],
            "confidence": round(float(suggestion["confidence"]), 3),
            "originalText": suggestion["originalText"],
            "suggestedText": suggestion["suggestedText"],
            "reason": suggestion["reason"],
            "requiresAudioReview": bool(suggestion["requiresAudioReview"]),
            "canBatchApply": bool(suggestion["canBatchApply"]),
            "status": "pending",
            "source": LLM_SOURCE,
            "sourceRunId": run_id,
            "startTime": segment.get("start"),
            "endTime": segment.get("end"),
        }
        corrections.append(correction)
        added.append(correction)
        existing_keys.add(key)
        next_index += 1

    if added:
        corrections_data["updatedAt"] = _now()
        _write_json(corrections_path, corrections_data)

    return {
        "addedCount": len(added),
        "skippedCount": len(skipped),
        "addedCorrections": added,
        "skippedCorrections": skipped,
    }


def is_safe_asr_correction(correction: dict[str, Any]) -> bool:
    return (
        correction.get("category") in SAFE_BATCH_CATEGORIES
        and float(correction.get("confidence") or 0.0) >= 0.95
        and correction.get("severity") == "low"
        and correction.get("requiresAudioReview") is False
        and correction.get("canBatchApply") is True
        and correction.get("status") == "pending"
    )


def _segment_map(approved: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {segment["id"]: segment for segment in approved.get("segments", []) if isinstance(segment, dict) and segment.get("id")}


def _replace_once(text: str, old: str, new: str) -> tuple[str, bool]:
    if old == "":
        return text, False
    if text == old:
        return new, text != new
    index = text.find(old)
    if index < 0:
        return text, False
    return text[:index] + new + text[index + len(old) :], old != new


def _apply_correction_to_approved(
    approved: dict[str, Any],
    correction: dict[str, Any],
    replacement_text: str | None = None,
) -> dict[str, Any] | None:
    segments = _segment_map(approved)
    segment = segments.get(str(correction.get("segmentId")))
    if segment is None:
        return None

    before = _text(segment.get("text"))
    after, changed = _replace_once(before, _text(correction.get("originalText")), replacement_text or _text(correction.get("suggestedText")))
    if not changed:
        return None

    segment["text"] = after
    approved["updatedAt"] = _now()
    return {
        "correctionId": correction["id"],
        "segmentId": segment["id"],
        "before": before,
        "after": after,
    }


def update_correction(output_dir: Path, correction_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    ensure_review_artifacts(output_dir)
    review_dir = _review_dir(Path(output_dir))
    corrections_data = _read_json(review_dir / "correction_suggestions.json")
    approved = _read_json(review_dir / "approved_transcript.json")
    status = str(payload.get("status") or "")
    if status not in CORRECTION_STATUSES:
        raise ValueError("Некорректный статус правки.")

    replacement_text = payload.get("suggestedText")
    changed_correction: dict[str, Any] | None = None
    edit_item = None
    for correction in corrections_data.get("corrections", []):
        if correction.get("id") != correction_id:
            continue
        correction["status"] = status
        if status == "modified":
            if not isinstance(replacement_text, str) or not replacement_text:
                raise ValueError("Для modified требуется suggestedText.")
            correction["suggestedText"] = replacement_text
        if status in {"accepted", "modified"}:
            edit_item = _apply_correction_to_approved(
                approved,
                correction,
                replacement_text if status == "modified" else None,
            )
        changed_correction = correction
        break

    if changed_correction is None:
        raise KeyError(correction_id)

    corrections_data["updatedAt"] = _now()
    _write_json(review_dir / "correction_suggestions.json", corrections_data)
    if edit_item is not None:
        _write_json(review_dir / "approved_transcript.json", approved)
    return {"correction": changed_correction, "edit": edit_item}


def apply_safe_asr_batch(output_dir: Path) -> dict[str, Any]:
    ensure_review_artifacts(output_dir)
    review_dir = _review_dir(Path(output_dir))
    corrections_data = _read_json(review_dir / "correction_suggestions.json")
    approved = _read_json(review_dir / "approved_transcript.json")
    batches_data = _read_json(review_dir / "edit_batches.json")

    applied = []
    skipped = []
    for correction in corrections_data.get("corrections", []):
        if not is_safe_asr_correction(correction):
            continue
        edit_item = _apply_correction_to_approved(approved, correction)
        if edit_item is None:
            skipped.append(correction["id"])
            continue
        correction["status"] = "accepted"
        applied.append(edit_item)

    batch = {
        "id": f"batch-{uuid.uuid4().hex[:12]}",
        "type": "safe_asr",
        "status": "applied",
        "createdAt": _now(),
        "appliedCount": len(applied),
        "skippedCorrectionIds": skipped,
        "items": applied,
    }
    if applied:
        batches_data.setdefault("batches", []).append(batch)
        batches_data["updatedAt"] = _now()
        corrections_data["updatedAt"] = _now()
        _write_json(review_dir / "approved_transcript.json", approved)
        _write_json(review_dir / "correction_suggestions.json", corrections_data)
        _write_json(review_dir / "edit_batches.json", batches_data)

    return batch


def rollback_batch(output_dir: Path, batch_id: str | None = None) -> dict[str, Any]:
    ensure_review_artifacts(output_dir)
    review_dir = _review_dir(Path(output_dir))
    approved = _read_json(review_dir / "approved_transcript.json")
    corrections_data = _read_json(review_dir / "correction_suggestions.json")
    batches_data = _read_json(review_dir / "edit_batches.json")
    batches = batches_data.get("batches", [])

    candidates = [batch for batch in batches if batch.get("status") == "applied"]
    if batch_id:
        candidates = [batch for batch in candidates if batch.get("id") == batch_id]
    if not candidates:
        raise KeyError(batch_id or "latest")

    batch = candidates[-1]
    segments = _segment_map(approved)
    conflicts = []
    rolled_back = []
    for item in reversed(batch.get("items", [])):
        segment = segments.get(item.get("segmentId"))
        if segment is None:
            conflicts.append({"segmentId": item.get("segmentId"), "reason": "segment_not_found"})
            continue
        current_text = _text(segment.get("text"))
        if current_text != item.get("after"):
            conflicts.append({"segmentId": segment["id"], "reason": "text_changed"})
            continue
        segment["text"] = item.get("before", current_text)
        rolled_back.append(item)

    if conflicts:
        batch["status"] = "rollback_failed"
        batch["conflicts"] = conflicts
    else:
        batch["status"] = "rolled_back"
        batch["rolledBackAt"] = _now()
        correction_ids = {item.get("correctionId") for item in rolled_back}
        for correction in corrections_data.get("corrections", []):
            if correction.get("id") in correction_ids:
                correction["status"] = "pending"
        corrections_data["updatedAt"] = _now()
        approved["updatedAt"] = _now()
        _write_json(review_dir / "approved_transcript.json", approved)
        _write_json(review_dir / "correction_suggestions.json", corrections_data)

    batches_data["updatedAt"] = _now()
    _write_json(review_dir / "edit_batches.json", batches_data)
    return {"batch": batch, "rolledBackCount": len(rolled_back), "conflicts": conflicts}


def update_entity(output_dir: Path, entity_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    ensure_review_artifacts(output_dir)
    review_dir = _review_dir(Path(output_dir))
    entities_data = _read_json(review_dir / "entity_annotations.json")
    changed_entity: dict[str, Any] | None = None

    for entity in entities_data.get("entities", []):
        if entity.get("id") != entity_id:
            continue
        if "status" in payload:
            status = str(payload["status"])
            if status not in ENTITY_STATUSES:
                raise ValueError("Некорректный статус сущности.")
            entity["status"] = status
        if "verificationStatus" in payload:
            verification_status = str(payload["verificationStatus"])
            if verification_status not in ENTITY_VERIFICATION_STATUSES:
                raise ValueError("Некорректный статус проверки сущности.")
            entity["verificationStatus"] = verification_status
        if "canonical" in payload:
            canonical = payload["canonical"]
            if canonical is not None and not isinstance(canonical, str):
                raise ValueError("canonical должен быть строкой.")
            entity["canonical"] = canonical
        changed_entity = entity
        break

    if changed_entity is None:
        raise KeyError(entity_id)

    entities_data["updatedAt"] = _now()
    _write_json(review_dir / "entity_annotations.json", entities_data)
    return {"entity": changed_entity}
