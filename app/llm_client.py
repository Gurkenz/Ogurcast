from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


class LLMClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    model: str
    temperature: float
    timeout_sec: float
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "LLMSettings":
        return cls(
            base_url=os.getenv("OGURCAST_LLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/"),
            model=os.getenv("OGURCAST_LLM_MODEL", "qwen3-8b").strip() or "qwen3-8b",
            temperature=_float_env("OGURCAST_LLM_TEMPERATURE", 0.15),
            timeout_sec=_float_env("OGURCAST_LLM_TIMEOUT_SEC", 120.0),
            api_key=os.getenv("OGURCAST_LLM_API_KEY") or None,
        )


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class OpenAICompatibleLLMClient:
    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings.from_env()

    def redacted_request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "baseUrl": self.settings.base_url,
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "timeoutSec": self.settings.timeout_sec,
            "hasApiKey": bool(self.settings.api_key),
            "messages": messages,
            "responseFormat": {"type": "json_object"},
        }

    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        try:
            with httpx.Client(timeout=self.settings.timeout_sec) as client:
                response = client.post(f"{self.settings.base_url}/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                raw_response = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMClientError(f"LM Studio вернул HTTP {exc.response.status_code}.") from exc
        except httpx.TimeoutException as exc:
            raise LLMClientError("LM Studio не ответил до истечения timeout.") from exc
        except httpx.RequestError as exc:
            raise LLMClientError(f"LM Studio недоступен: {exc}") from exc
        except ValueError as exc:
            raise LLMClientError("LM Studio вернул невалидный JSON.") from exc

        try:
            content = raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("LM Studio response не содержит choices[0].message.content.") from exc

        if isinstance(content, str):
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError as exc:
                raise LLMClientError("Qwen3 вернул текст вместо строгого JSON.") from exc
        elif isinstance(content, dict):
            parsed_content = content
        else:
            raise LLMClientError("Qwen3 вернул неподдерживаемый формат content.")

        return {"rawResponse": raw_response, "content": parsed_content}
