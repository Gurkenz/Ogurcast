from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.llm_contract import LLMCorrectionResponse


def _valid_payload() -> dict:
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
                "reason": "ASR исказил название.",
                "requiresAudioReview": False,
                "canBatchApply": False,
            }
        ],
    }


def test_llm_contract_accepts_valid_payload() -> None:
    parsed = LLMCorrectionResponse.model_validate(_valid_payload())

    assert parsed.model == "qwen3-8b"
    assert parsed.corrections[0].category == "ASR_ERROR"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category", "STYLE_REWRITE"),
        ("severity", "critical"),
        ("confidence", 1.2),
    ],
)
def test_llm_contract_rejects_invalid_fields(field: str, value: object) -> None:
    payload = _valid_payload()
    payload["corrections"][0][field] = value

    with pytest.raises(ValidationError):
        LLMCorrectionResponse.model_validate(payload)


def test_llm_contract_rejects_model_control_fields() -> None:
    payload = _valid_payload()
    payload["corrections"][0]["id"] = "corr-owned-by-model"

    with pytest.raises(ValidationError):
        LLMCorrectionResponse.model_validate(payload)
