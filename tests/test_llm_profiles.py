from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.llm_profiles import LLMProfile, profile_hash, read_local_profiles, render_messages, save_local_profile


def _profile_payload() -> dict[str, object]:
    return {
        "profileId": "local-asr-correction",
        "version": 1,
        "label": "ASR correction",
        "stageDescription": "Точечные правки ASR.",
        "systemInstruction": "Верни JSON.",
        "userPayloadTemplate": '{"schema": {output_schema_json}, "input": {input_json}}',
        "schemaNotes": "Только LLMCorrectionResponse.",
        "defaultModel": "qwen3-8b",
        "temperature": 0.1,
        "timeoutSec": 120,
        "maxInputChars": 18000,
    }


def test_profile_validation_requires_payload_placeholders() -> None:
    payload = _profile_payload()
    payload["stage"] = "asr_correction"
    payload["userPayloadTemplate"] = '{"input": {input_json}}'

    with pytest.raises(ValueError):
        LLMProfile.model_validate(payload)


def test_save_local_profile_roundtrip(monkeypatch, tmp_path: Path) -> None:
    import app.llm_profiles as profiles

    local_path = tmp_path / "llm_profiles.local.json"
    monkeypatch.setattr(profiles, "LOCAL_PROFILE_PATH", local_path)

    saved = save_local_profile("asr_correction", _profile_payload())
    loaded = read_local_profiles(local_path)

    assert saved.profileId == "local-asr-correction"
    assert loaded["asr_correction"].systemInstruction == "Верни JSON."
    assert json.loads(local_path.read_text(encoding="utf-8"))["version"] == 1


def test_render_messages_and_hash_are_reproducible() -> None:
    payload = _profile_payload()
    payload["stage"] = "asr_correction"
    profile = LLMProfile.model_validate(payload)
    input_payload = {"version": 1, "segments": [{"id": "seg-000001", "text": "текст"}]}

    messages = render_messages(profile, input_payload, "qwen3-8b")

    assert messages[0]["role"] == "system"
    assert "seg-000001" in messages[1]["content"]
    assert profile_hash(profile, messages) == profile_hash(profile, messages)
