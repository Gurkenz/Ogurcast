from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from app import jobs
from app.config import PROJECT_ROOT


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_output_dir(path: Path) -> None:
    segments = [{"start": 0.0, "end": 1.0, "text": "Текст", "speaker": "SPEAKER_00", "words": []}]
    _write_json(path / "segments.json", segments)
    _write_json(path / "words.json", [])
    _write_json(path / "result_raw.json", {"segments": segments})


def _tmp_root() -> Path:
    tmp_root = PROJECT_ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    return tmp_root


def test_successful_diarization_queues_auto_audit(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        jobs,
        "start_auto_llm_audit",
        lambda **kwargs: calls.append(kwargs) or {"run_id": "llm-test", "status": "queued"},
    )
    with tempfile.TemporaryDirectory(dir=_tmp_root()) as tmp:
        output_dir = Path(tmp)
        _seed_output_dir(output_dir)
        jobs._after_successful_job(
            "job-1",
            {"output_dir": str(output_dir), "metadata": {"diarization_status": "done"}},
        )

        assert calls
        assert calls[0]["job_id"] == "job-1"


def test_failed_diarization_records_technical_flag_without_auto_audit(monkeypatch) -> None:
    monkeypatch.setattr(
        jobs,
        "start_auto_llm_audit",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("auto audit should not start")),
    )
    with tempfile.TemporaryDirectory(dir=_tmp_root()) as tmp:
        output_dir = Path(tmp)
        _seed_output_dir(output_dir)
        jobs._after_successful_job(
            "job-1",
            {"output_dir": str(output_dir), "metadata": {"diarization_status": "failed"}},
        )

        flags = _read_json(output_dir / "review" / "technical_flags.json")["flags"]
        assert any(item["flag"] == "diarization_not_done" for item in flags)
