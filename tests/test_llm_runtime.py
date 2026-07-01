from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from app.config import PROJECT_ROOT
from app.llm_client import LLMClientError, LLMSettings
from app.llm_runtime import ensure_llm_runtime


def _settings(exe: Path, model: Path) -> LLMSettings:
    return LLMSettings(
        base_url="http://127.0.0.1:8080/v1",
        model=model.name,
        temperature=0.05,
        timeout_sec=180,
        provider="llamacpp",
        llama_server_exe=str(exe),
        llama_model_path=str(model),
    )


def test_llamacpp_runtime_starts_when_not_reachable(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as tmp:
        root = Path(tmp)
        exe = root / "llama-server.exe"
        model = root / "model.gguf"
        exe.write_bytes(b"exe")
        model.write_bytes(b"gguf")
        calls: list[tuple[Path, Path]] = []
        statuses: list[dict[str, Any]] = [
            {"reachable": False, "modelLoaded": False, "availableModels": [], "error": "connection refused"},
            {"reachable": True, "modelLoaded": True, "availableModels": [model.name], "error": None},
        ]

        import app.llm_runtime as runtime

        monkeypatch.setattr(runtime, "_runtime_status", lambda settings: statuses.pop(0) if statuses else {"reachable": True, "modelLoaded": True, "availableModels": [model.name], "error": None})
        monkeypatch.setattr(runtime, "_start_llama_server", lambda settings, exe_path, model_path: calls.append((exe_path, model_path)))
        monkeypatch.setattr(runtime.time, "sleep", lambda delay: None)

        result = ensure_llm_runtime(_settings(exe, model))

        assert result["status"] == "started"
        assert calls == [(exe, model)]


def test_llamacpp_runtime_requires_repo_local_exe_and_model() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as tmp:
        root = Path(tmp)
        exe = root / "missing-llama-server.exe"
        model = root / "model.gguf"
        model.write_bytes(b"gguf")

        with pytest.raises(LLMClientError):
            ensure_llm_runtime(_settings(exe, model))


def test_non_llamacpp_runtime_is_external() -> None:
    settings = LLMSettings(
        base_url="http://127.0.0.1:1234/v1",
        model="qwen3-8b",
        temperature=0.1,
        timeout_sec=120,
        provider="lmstudio",
    )

    assert ensure_llm_runtime(settings)["status"] == "external"
