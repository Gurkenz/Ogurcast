from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


CorrectionCategory = Literal["ASR_ERROR", "TYPO", "PUNCTUATION", "FILLER_GARBAGE", "NEEDS_LISTENING"]
CorrectionSeverity = Literal["low", "medium", "high"]


class LLMCorrectionSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segmentId: str = Field(min_length=1)
    category: CorrectionCategory
    severity: CorrectionSeverity
    confidence: float = Field(ge=0.0, le=1.0)
    originalText: str = Field(min_length=1)
    suggestedText: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    requiresAudioReview: bool
    canBatchApply: bool

    @field_validator("segmentId", "originalText", "suggestedText", "reason")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped


class LLMCorrectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    model: str = Field(min_length=1)
    corrections: list[LLMCorrectionSuggestion]

    @field_validator("model")
    @classmethod
    def _strip_model(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("model must not be empty")
        return stripped
