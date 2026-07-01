from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


CorrectionCategory = Literal["ASR_ERROR", "TYPO", "PUNCTUATION", "FILLER_GARBAGE", "NEEDS_LISTENING"]
CorrectionSeverity = Literal["low", "medium", "high"]
AuditCategory = Literal[
    "ASR_OBVIOUS",
    "ASR_PUNCTUATION",
    "ASR_ENTITY_UNCERTAIN",
    "ASR_LOW_CONFIDENCE",
    "ASR_LOOP",
    "DIARIZATION_WRONG_SPEAKER",
    "DIARIZATION_BOUNDARY",
    "DIARIZATION_INTRASEGMENT_SHIFT",
    "DIARIZATION_EMBEDDED_REPLY",
    "DIARIZATION_MICROSEGMENT",
    "SERVICE_FRAGMENT",
    "SOURCE_CONFLICT",
    "AUDIO_REVIEW",
    "GLOSSARY_ENTITY",
    "TECHNICAL_FLAG",
]
AuditAction = Literal[
    "replace_text",
    "add_punctuation",
    "change_case",
    "split_segment",
    "merge_segments",
    "move_text_to_previous_segment",
    "move_text_to_next_segment",
    "relabel_speaker",
    "mark_service_fragment",
    "mark_audio_review",
    "add_glossary_entry",
    "flag_source_conflict",
    "flag_technical_issue",
    "keep_unchanged",
]
AuditSeverity = Literal["low", "medium", "high", "critical"]
AuditRisk = Literal["low", "medium", "high"]
AuditEvidenceType = Literal[
    "context",
    "word_score",
    "avg_logprob",
    "speaker_shift",
    "source_conflict",
    "metadata",
    "entity_evidence",
    "pattern",
]
GlossaryType = Literal[
    "person",
    "organization",
    "project",
    "event",
    "drug",
    "medical_term",
    "technical_term",
    "abbreviation",
    "number",
    "date",
    "other",
]
GlossaryStatus = Literal[
    "confirmed_by_context",
    "confirmed_by_evidence",
    "requires_audio_review",
    "requires_external_check",
    "uncertain",
]


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


class AuditEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: AuditEvidenceType
    value: str = Field(min_length=1)

    @field_validator("value")
    @classmethod
    def _strip_value(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped


class LLMAuditSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segmentId: str = Field(min_length=1)
    involvedSegmentIds: list[str] = Field(default_factory=list)
    category: AuditCategory
    action: AuditAction
    severity: AuditSeverity
    risk: AuditRisk
    confidence: float = Field(ge=0.0, le=1.0)
    canAutoApply: bool
    requiresAudioReview: bool
    originalText: str = ""
    suggestedText: str = ""
    reason: str = Field(min_length=1)
    evidence: list[AuditEvidence] = Field(default_factory=list)

    @field_validator("segmentId", "reason")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped

    @field_validator("originalText", "suggestedText")
    @classmethod
    def _strip_optional_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("involvedSegmentIds")
    @classmethod
    def _strip_segment_ids(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]


class LLMSpeakerMapSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speakerLabel: str = Field(min_length=1)
    proposedRole: str = ""
    proposedName: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    requiresReview: bool

    @field_validator("speakerLabel", "reason")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped

    @field_validator("proposedRole", "proposedName")
    @classmethod
    def _strip_optional_text(cls, value: str) -> str:
        return value.strip()


class LLMGlossaryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str = Field(min_length=1)
    type: GlossaryType
    canonical: str = Field(min_length=1)
    status: GlossaryStatus
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)

    @field_validator("term", "canonical", "reason")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped


class LLMTechnicalFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flag: str = Field(min_length=1)
    severity: AuditSeverity
    risk: AuditRisk
    reason: str = Field(min_length=1)
    recommendedAction: str = Field(min_length=1)

    @field_validator("flag", "reason", "recommendedAction")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped


class LLMAuditSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hasCriticalIssues: bool
    autoApplyCount: int = Field(ge=0)
    audioReviewCount: int = Field(ge=0)
    diarizationIssueCount: int = Field(ge=0)
    asrIssueCount: int = Field(ge=0)
    entityIssueCount: int = Field(ge=0)


class LLMAuditResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str | int
    task: Literal["diarized_asr_audit"]
    chunkId: str = Field(min_length=1)
    suggestions: list[LLMAuditSuggestion]
    speakerMapSuggestions: list[LLMSpeakerMapSuggestion]
    glossary: list[LLMGlossaryEntry]
    technicalFlags: list[LLMTechnicalFlag]
    summary: LLMAuditSummary

    @field_validator("version")
    @classmethod
    def _strip_version(cls, value: str | int) -> str | int:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("version must not be empty")
            return stripped
        return value

    @field_validator("chunkId")
    @classmethod
    def _strip_chunk_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("chunkId must not be empty")
        return stripped
