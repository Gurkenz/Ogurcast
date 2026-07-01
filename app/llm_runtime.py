from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.file_utils import ensure_inside_project
from app.llm_client import LLMClientError, LLMSettings


_MANAGED_PROCESS: subprocess.Popen | None = None


def ensure_llm_runtime(settings: LLMSettings) -> dict[str, Any]:
    if settings.provider != "llamacpp":
        return {"provider": settings.provider, "managed": False, "status": "external"}

    status = _runtime_status(settings)
    if status["reachable"] and status["modelLoaded"]:
        return {"provider": settings.provider, "managed": True, "status": "ready", **status}

    exe_path = ensure_inside_project(Path(settings.llama_server_exe))
    model_path = ensure_inside_project(Path(settings.llama_model_path))
    if not exe_path.is_file():
        raise LLMClientError(f"llama-server.exe не найден: {exe_path}")
    if not model_path.is_file():
        raise LLMClientError(f"GGUF-модель не найдена: {model_path}")

    _start_llama_server(settings, exe_path, model_path)
    status = _wait_until_ready(settings, timeout_sec=45.0)
    if not status["reachable"]:
        raise LLMClientError(str(status.get("error") or "llama-server не поднялся."))
    if not status["modelLoaded"]:
        available = ", ".join(status.get("availableModels") or [])
        suffix = f" Доступно: {available}." if available else ""
        raise LLMClientError(f"llama-server поднялся, но модель не подтверждена: {settings.model}.{suffix}")
    return {"provider": settings.provider, "managed": True, "status": "started", **status}


def _start_llama_server(settings: LLMSettings, exe_path: Path, model_path: Path) -> None:
    global _MANAGED_PROCESS

    if _MANAGED_PROCESS is not None and _MANAGED_PROCESS.poll() is None:
        return

    parsed = urlparse(settings.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 8080)
    command = [
        str(exe_path),
        "--model",
        str(model_path),
        "--host",
        host,
        "--port",
        port,
        "--ctx-size",
        "8192",
        "--parallel",
        "1",
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    _MANAGED_PROCESS = subprocess.Popen(
        command,
        cwd=str(exe_path.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def _wait_until_ready(settings: LLMSettings, timeout_sec: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_status: dict[str, Any] = {"reachable": False, "modelLoaded": False}
    while time.monotonic() < deadline:
        last_status = _runtime_status(settings)
        if last_status["reachable"] and last_status["modelLoaded"]:
            return last_status
        time.sleep(1.0)
    return last_status


def _runtime_status(settings: LLMSettings) -> dict[str, Any]:
    health_url = _health_url(settings.base_url)
    models_url = f"{settings.base_url}/models"
    try:
        with httpx.Client(timeout=5.0) as client:
            health = client.get(health_url)
            health.raise_for_status()
            models_response = client.get(models_url)
            models_response.raise_for_status()
            models_payload = models_response.json()
    except httpx.HTTPStatusError as exc:
        return {
            "reachable": True,
            "modelLoaded": False,
            "configuredModel": settings.model,
            "availableModels": [],
            "error": f"llama-server вернул HTTP {exc.response.status_code}.",
        }
    except (httpx.RequestError, ValueError) as exc:
        return {
            "reachable": False,
            "modelLoaded": False,
            "configuredModel": settings.model,
            "availableModels": [],
            "error": f"llama-server недоступен: {exc}",
        }

    models = _model_ids(models_payload)
    model_loaded = settings.model in models or len(models) == 1
    return {
        "reachable": True,
        "modelLoaded": model_loaded,
        "configuredModel": settings.model,
        "availableModels": models,
        "currentModel": models[0] if len(models) == 1 else None,
        "error": None,
    }


def _health_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    root = parsed._replace(path=path or "", params="", query="", fragment="").geturl().rstrip("/")
    return f"{root}/health"


def _model_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
            ids.append(item["id"].strip())
    return ids
