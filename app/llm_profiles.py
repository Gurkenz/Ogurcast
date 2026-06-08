from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import PROJECT_ROOT
from app.llm_client import LLMSettings


LOCAL_PROFILE_PATH = PROJECT_ROOT / "config" / "llm_profiles.local.json"
SUPPORTED_STAGES = {"asr_correction"}

ASR_CORRECTION_SCHEMA: dict[str, Any] = {
    "version": 1,
    "model": "model-name",
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

DEFAULT_ASR_SYSTEM_INSTRUCTION = (
    "Ты внутренний редактор русскоязычной ASR-расшифровки Ogurcast. "
    "Верни только валидный JSON, без markdown и пояснений вне JSON. "
    "Корневой объект ответа должен содержать только поля version, model, corrections. "
    "Не возвращай task, output_schema или input. Не повторяй входной payload. "
    "Не переписывай сегменты целиком. Не меняй speaker, start/end и segmentId. "
    "Предлагай только точечные CorrectionSuggestion для очевидных ASR-ошибок, опечаток, пунктуации, "
    "мусора распознавания или мест, требующих прослушивания. "
    "Используй adjacentContext, speakerTurnText и lowConfidenceWords для контекстной проверки. "
    "Если контекст ясно указывает на исправление, верни реальную замену, например медицинский термин вместо фонетической ошибки. "
    "Если уверенности нет, ставь requiresAudioReview=true и canBatchApply=false, но suggestedText не должен повторять originalText. "
    "Запрещено добавлять поля id, status, source, sourceRunId, startTime, endTime."
)

DEFAULT_USER_PAYLOAD_TEMPLATE = (
    "Задача: найди точечные correction suggestions для ASR-расшифровки.\n\n"
    "Верни ответ строго в этом JSON-формате, без дополнительных полей:\n"
    "{output_schema_json}\n\n"
    "Входные данные для анализа:\n"
    "{input_json}"
)


class LLMProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str = Field(min_length=1)
    profileId: str = Field(min_length=1)
    version: int = Field(ge=1)
    label: str = Field(min_length=1)
    stageDescription: str = Field(min_length=1)
    systemInstruction: str = Field(min_length=1)
    userPayloadTemplate: str = Field(min_length=1)
    schemaNotes: str = Field(min_length=1)
    defaultModel: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    timeoutSec: float | None = Field(default=None, gt=0.0, le=600.0)
    maxInputChars: int = Field(default=18000, ge=1000, le=200000)

    @field_validator("stage")
    @classmethod
    def _supported_stage(cls, value: str) -> str:
        stripped = value.strip()
        if stripped not in SUPPORTED_STAGES:
            raise ValueError("unsupported LLM stage")
        return stripped

    @field_validator(
        "profileId",
        "label",
        "stageDescription",
        "systemInstruction",
        "userPayloadTemplate",
        "schemaNotes",
        "defaultModel",
    )
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped

    @field_validator("userPayloadTemplate")
    @classmethod
    def _requires_input_placeholder(cls, value: str) -> str:
        if "{input_json}" not in value:
            raise ValueError("userPayloadTemplate must contain {input_json}")
        if "{output_schema_json}" not in value:
            raise ValueError("userPayloadTemplate must contain {output_schema_json}")
        return value


def default_profiles() -> dict[str, LLMProfile]:
    env_settings = LLMSettings.from_env()
    return {
        "asr_correction": LLMProfile(
            stage="asr_correction",
            profileId="default-asr-correction",
            version=1,
            label="ASR correction",
            stageDescription="Точечные правки ASR без прямого изменения transcript.",
            systemInstruction=DEFAULT_ASR_SYSTEM_INSTRUCTION,
            userPayloadTemplate=DEFAULT_USER_PAYLOAD_TEMPLATE,
            schemaNotes=(
                "LLM возвращает только объект LLMCorrectionResponse. "
                "segmentId должен ссылаться на входной segment. "
                "Сервер сам добавляет id/status/source/timestamps."
            ),
            defaultModel=env_settings.model,
            temperature=env_settings.temperature,
            timeoutSec=env_settings.timeout_sec,
            maxInputChars=18000,
        )
    }


def read_local_profiles(path: Path | None = None) -> dict[str, LLMProfile]:
    path = path or LOCAL_PROFILE_PATH
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", {}) if isinstance(payload, dict) else {}
    if not isinstance(profiles, dict):
        raise ValueError("llm_profiles.local.json поврежден.")
    return {
        stage: LLMProfile.model_validate(profile)
        for stage, profile in profiles.items()
        if isinstance(profile, dict)
    }


def list_profiles() -> dict[str, Any]:
    defaults = default_profiles()
    local = read_local_profiles()
    effective = {stage: local.get(stage, profile) for stage, profile in defaults.items()}
    return {
        "version": 1,
        "localPath": str(LOCAL_PROFILE_PATH),
        "stages": sorted(defaults),
        "defaults": {stage: profile.model_dump(mode="json") for stage, profile in defaults.items()},
        "local": {stage: profile.model_dump(mode="json") for stage, profile in local.items()},
        "effective": {stage: profile.model_dump(mode="json") for stage, profile in effective.items()},
    }


def get_profile(stage: str) -> LLMProfile:
    stage = stage.strip()
    defaults = default_profiles()
    if stage not in defaults:
        raise KeyError(stage)
    return read_local_profiles().get(stage, defaults[stage])


def save_local_profile(stage: str, payload: dict[str, Any]) -> LLMProfile:
    if stage not in SUPPORTED_STAGES:
        raise KeyError(stage)
    profile_payload = dict(payload)
    profile_payload["stage"] = stage
    if not profile_payload.get("profileId"):
        profile_payload["profileId"] = f"local-{stage}"
    if not profile_payload.get("version"):
        profile_payload["version"] = 1
    profile = LLMProfile.model_validate(profile_payload)

    profiles = read_local_profiles()
    profiles[stage] = profile
    LOCAL_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_PROFILE_PATH.write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {key: item.model_dump(mode="json") for key, item in sorted(profiles.items())},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return profile


def render_messages(profile: LLMProfile, input_payload: dict[str, Any], model: str) -> list[dict[str, str]]:
    schema = dict(ASR_CORRECTION_SCHEMA)
    schema["model"] = model
    user_content = profile.userPayloadTemplate.replace(
        "{output_schema_json}",
        json.dumps(schema, ensure_ascii=False, indent=2),
    ).replace(
        "{input_json}",
        json.dumps(input_payload, ensure_ascii=False, indent=2),
    )
    return [
        {"role": "system", "content": profile.systemInstruction},
        {"role": "user", "content": user_content},
    ]


def profile_hash(profile: LLMProfile, messages: list[dict[str, str]]) -> str:
    payload = {
        "profile": profile.model_dump(mode="json"),
        "messages": messages,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()
