from __future__ import annotations

import json
import os
import re
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

    def model_status(self) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        try:
            with httpx.Client(timeout=min(self.settings.timeout_sec, 10.0)) as client:
                response = client.get(f"{self.settings.base_url}/models", headers=headers)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            return {
                "reachable": True,
                "modelLoaded": False,
                "configuredModel": self.settings.model,
                "availableModels": [],
                "error": f"LM Studio /models вернул HTTP {exc.response.status_code}.",
            }
        except httpx.TimeoutException:
            return {
                "reachable": False,
                "modelLoaded": False,
                "configuredModel": self.settings.model,
                "availableModels": [],
                "error": "LM Studio /models не ответил до истечения timeout.",
            }
        except (httpx.RequestError, ValueError) as exc:
            return {
                "reachable": False,
                "modelLoaded": False,
                "configuredModel": self.settings.model,
                "availableModels": [],
                "error": f"LM Studio /models недоступен: {exc}",
            }

        models = _model_ids(payload)
        return {
            "reachable": True,
            "modelLoaded": self.settings.model in models,
            "configuredModel": self.settings.model,
            "availableModels": models,
            "currentModel": models[0] if len(models) == 1 else None,
            "error": None,
        }

    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
                response = _post_completion_with_json_mode_retry(client, self.settings.base_url, payload, headers)
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
            parsed_content = _parse_json_content(content)
        elif isinstance(content, dict):
            parsed_content = content
        else:
            raise LLMClientError("LLM вернул неподдерживаемый формат content.")

        return {"rawResponse": raw_response, "content": parsed_content}


def _model_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    ids = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
            ids.append(item["id"].strip())
    return ids


def _post_completion_with_json_mode_retry(
    client: httpx.Client,
    base_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response:
    response = client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
    if response.status_code == 400 and "response_format" in payload:
        fallback_payload = dict(payload)
        fallback_payload.pop("response_format", None)
        response = client.post(f"{base_url}/chat/completions", json=fallback_payload, headers=headers)
    response.raise_for_status()
    return response


def _parse_json_content(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
        if match is None:
            raise LLMClientError("LLM вернул текст вместо строгого JSON.")
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM вернул текст вместо строгого JSON.") from exc
    if not isinstance(parsed, dict):
        raise LLMClientError("LLM JSON должен быть объектом.")
    return parsed
