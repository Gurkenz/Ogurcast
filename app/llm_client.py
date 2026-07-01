from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
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
    provider: str = "llamacpp"
    llama_server_exe: str = r"Z:\Ogurcast\tools\llama-server.exe"
    llama_model_path: str = r"Z:\Ogurcast\models\llm\qwen3-8b-q4_k_m.gguf"

    @classmethod
    def from_env(cls) -> "LLMSettings":
        raw_provider = os.getenv("OGURCAST_LLM_PROVIDER")
        provider = (raw_provider or "llamacpp").strip().casefold() or "llamacpp"
        llama_model_path = os.getenv("OGURCAST_LLAMA_MODEL_PATH", r"Z:\Ogurcast\models\llm\qwen3-8b-q4_k_m.gguf").strip()
        default_base_url = "http://127.0.0.1:8080/v1" if provider == "llamacpp" else "http://127.0.0.1:1234/v1"
        default_model = Path(llama_model_path).name if provider == "llamacpp" else "qwen3-8b"
        base_url = os.getenv("OGURCAST_LLM_BASE_URL", default_base_url)
        model = os.getenv("OGURCAST_LLM_MODEL", default_model)
        if raw_provider is None and provider == "llamacpp":
            base_url = default_base_url
            model = default_model
        return cls(
            base_url=base_url.rstrip("/"),
            model=model.strip() or default_model,
            temperature=_float_env("OGURCAST_LLM_TEMPERATURE", 0.05),
            timeout_sec=_float_env("OGURCAST_LLM_TIMEOUT_SEC", 180.0),
            api_key=os.getenv("OGURCAST_LLM_API_KEY") or None,
            provider=provider,
            llama_server_exe=os.getenv("OGURCAST_LLAMA_SERVER_EXE", r"Z:\Ogurcast\tools\llama-server.exe").strip(),
            llama_model_path=llama_model_path,
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
            "provider": self.settings.provider,
            "baseUrl": self.settings.base_url,
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "timeoutSec": self.settings.timeout_sec,
            "llamaServerExe": self.settings.llama_server_exe if self.settings.provider == "llamacpp" else None,
            "llamaModelPath": self.settings.llama_model_path if self.settings.provider == "llamacpp" else None,
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
                "provider": self.settings.provider,
                "configuredModel": self.settings.model,
                "availableModels": [],
                "error": f"LLM endpoint /models вернул HTTP {exc.response.status_code}.",
            }
        except httpx.TimeoutException:
            return {
                "reachable": False,
                "modelLoaded": False,
                "provider": self.settings.provider,
                "configuredModel": self.settings.model,
                "availableModels": [],
                "error": "LLM endpoint /models не ответил до истечения timeout.",
            }
        except (httpx.RequestError, ValueError) as exc:
            return {
                "reachable": False,
                "modelLoaded": False,
                "provider": self.settings.provider,
                "configuredModel": self.settings.model,
                "availableModels": [],
                "error": f"LLM endpoint /models недоступен: {exc}",
            }

        models = _model_ids(payload)
        model_loaded = self.settings.model in models
        current_model = models[0] if len(models) == 1 else None
        if self.settings.provider == "llamacpp" and len(models) == 1:
            model_loaded = True
        return {
            "reachable": True,
            "modelLoaded": model_loaded,
            "provider": self.settings.provider,
            "configuredModel": self.settings.model,
            "availableModels": models,
            "currentModel": current_model,
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
            raise LLMClientError(f"LLM endpoint вернул HTTP {exc.response.status_code}.") from exc
        except httpx.TimeoutException as exc:
            raise LLMClientError("LLM endpoint не ответил до истечения timeout.") from exc
        except httpx.RequestError as exc:
            raise LLMClientError(f"LLM endpoint недоступен: {exc}") from exc
        except ValueError as exc:
            raise LLMClientError("LLM endpoint вернул невалидный JSON.") from exc

        try:
            content = raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("LLM endpoint response не содержит choices[0].message.content.") from exc

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
