from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from app.config import PROJECT_ROOT
from app.review_artifacts import (
    apply_safe_asr_batch,
    ensure_review_artifacts,
    get_entities,
    reassign_segment_speaker,
    rollback_batch,
    update_correction,
    update_entity,
    update_speaker,
)


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReviewArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp_root = PROJECT_ROOT / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=tmp_root)
        self.output_dir = Path(self.tmp.name)
        self.segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": " Привет   мир",
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "Привет", "start": 0.0, "end": 0.5, "score": 0.99, "speaker": "SPEAKER_00"},
                    {"word": "мир", "start": 0.6, "end": 1.0, "score": 0.70, "speaker": "SPEAKER_00"},
                ],
            },
            {
                "start": 2.0,
                "end": 5.0,
                "text": "Компания «Сириус Биотех» встретила Иван Петров.",
                "speaker": "SPEAKER_01",
                "words": [
                    {"word": "Компания", "start": 2.0, "end": 2.4, "score": 0.96, "speaker": "SPEAKER_01"},
                    {"word": "Сириус", "start": 2.5, "end": 2.9, "score": 0.98, "speaker": "SPEAKER_01"},
                ],
            },
        ]
        _write_json(self.output_dir / "segments.json", self.segments)
        _write_json(self.output_dir / "words.json", [word for segment in self.segments for word in segment["words"]])
        _write_json(self.output_dir / "result_raw.json", {"segments": self.segments, "raw": True})

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_generates_review_artifacts_from_existing_outputs(self) -> None:
        bundle = ensure_review_artifacts(self.output_dir)

        self.assertTrue((self.output_dir / "review" / "transcript_normalized.json").is_file())
        self.assertTrue((self.output_dir / "review" / "correction_suggestions.json").is_file())
        self.assertTrue((self.output_dir / "review" / "audio_flags.json").is_file())
        self.assertEqual(bundle["transcript"]["segments"][0]["id"], "seg-000001")
        self.assertEqual(bundle["transcript"]["segments"][0]["words"][0]["id"], "word-000001-0001")
        self.assertEqual(bundle["transcript"]["segments"][0]["text"], "Привет мир")
        self.assertEqual(bundle["approvedTranscript"]["segments"][0]["text"], "Привет мир")

        corrections = bundle["corrections"]["corrections"]
        self.assertEqual(corrections, [])
        flags = bundle["audioFlags"]["flags"]
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["category"], "LOW_ASR_CONFIDENCE")
        self.assertEqual(flags[0]["text"], "мир")
        self.assertEqual(len(bundle["speakerTurns"]["turns"]), 2)

        entities = get_entities(self.output_dir)["entities"]
        surfaces = {entity["surface"] for entity in entities}
        self.assertIn("Сириус Биотех", surfaces)

    def test_safe_batch_apply_and_rollback_preserve_raw_artifacts(self) -> None:
        raw_hashes = {
            name: _sha256(self.output_dir / name)
            for name in ("result_raw.json", "segments.json", "words.json")
        }
        ensure_review_artifacts(self.output_dir)
        correction = {
            "version": 1,
            "updatedAt": "test",
            "corrections": [
                {
                    "id": "corr-000001",
                    "segmentId": "seg-000001",
                    "category": "TYPO",
                    "severity": "low",
                    "confidence": 0.99,
                    "originalText": "Привет",
                    "suggestedText": "Здравствуйте",
                    "reason": "test",
                    "requiresAudioReview": False,
                    "canBatchApply": True,
                    "status": "pending",
                }
            ],
        }
        _write_json(self.output_dir / "review" / "correction_suggestions.json", correction)

        batch = apply_safe_asr_batch(self.output_dir)
        self.assertEqual(batch["appliedCount"], 1)

        approved = _read_json(self.output_dir / "review" / "approved_transcript.json")
        self.assertEqual(approved["segments"][0]["text"], "Здравствуйте мир")
        corrections = _read_json(self.output_dir / "review" / "correction_suggestions.json")["corrections"]
        safe = [item for item in corrections if item["category"] == "TYPO"][0]
        self.assertEqual(safe["status"], "accepted")

        rollback = rollback_batch(self.output_dir, batch["id"])
        self.assertEqual(rollback["rolledBackCount"], 1)
        approved = _read_json(self.output_dir / "review" / "approved_transcript.json")
        self.assertEqual(approved["segments"][0]["text"], "Привет мир")
        corrections = _read_json(self.output_dir / "review" / "correction_suggestions.json")["corrections"]
        safe = [item for item in corrections if item["category"] == "TYPO"][0]
        self.assertEqual(safe["status"], "pending")

        self.assertEqual(
            raw_hashes,
            {name: _sha256(self.output_dir / name) for name in raw_hashes},
        )

    def test_patch_correction_updates_review_state_only(self) -> None:
        ensure_review_artifacts(self.output_dir)
        correction = {
            "id": "corr-000001",
            "segmentId": "seg-000001",
            "category": "TYPO",
            "severity": "medium",
            "confidence": 0.9,
            "originalText": "Привет",
            "suggestedText": "Здравствуйте",
            "reason": "test",
            "requiresAudioReview": False,
            "canBatchApply": False,
            "status": "pending",
        }
        _write_json(
            self.output_dir / "review" / "correction_suggestions.json",
            {"version": 1, "updatedAt": "test", "corrections": [correction]},
        )

        result = update_correction(self.output_dir, correction["id"], {"status": "modified", "suggestedText": "Здравствуйте"})
        self.assertEqual(result["correction"]["status"], "modified")

        approved = _read_json(self.output_dir / "review" / "approved_transcript.json")
        self.assertEqual(approved["segments"][0]["text"], "Здравствуйте мир")
        with self.assertRaises(ValueError):
            update_correction(self.output_dir, correction["id"], {"status": "unsafe"})

    def test_entity_update_validates_statuses(self) -> None:
        ensure_review_artifacts(self.output_dir)
        entity = get_entities(self.output_dir)["entities"][0]

        result = update_entity(
            self.output_dir,
            entity["id"],
            {
                "status": "accepted",
                "verificationStatus": "manual_confirmed",
                "canonical": "Сириус Биотех",
            },
        )

        self.assertEqual(result["entity"]["status"], "accepted")
        self.assertEqual(result["entity"]["verificationStatus"], "manual_confirmed")
        with self.assertRaises(ValueError):
            update_entity(self.output_dir, entity["id"], {"verificationStatus": "bad"})

    def test_speaker_turns_and_manual_reassignment_are_review_only(self) -> None:
        raw_hash = _sha256(self.output_dir / "segments.json")
        bundle = ensure_review_artifacts(self.output_dir)
        self.assertEqual([turn["speaker"] for turn in bundle["speakerTurns"]["turns"]], ["SPEAKER_00", "SPEAKER_01"])

        update_speaker(self.output_dir, "SPEAKER_00", {"displayName": "Ведущий", "verificationStatus": "manual_confirmed"})
        reassigned = reassign_segment_speaker(self.output_dir, "seg-000002", "SPEAKER_00")

        self.assertEqual(len(reassigned["speakerTurns"]), 1)
        transcript = _read_json(self.output_dir / "review" / "transcript_normalized.json")
        self.assertEqual(transcript["segments"][1]["speaker"], "SPEAKER_00")
        self.assertEqual(_sha256(self.output_dir / "segments.json"), raw_hash)


if __name__ == "__main__":
    unittest.main()
