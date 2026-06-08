from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, UPLOADS_DIR
from app import main


client = TestClient(main.app)


def test_llm_postprocess_rejects_unfinished_job(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_job", lambda job_id: {"id": job_id, "status": "running", "output_dir": None})

    response = client.post("/api/jobs/job-1/llm/postprocess")

    assert response.status_code == 400
    assert "Review доступен только после завершения задачи" in response.json()["detail"]


def test_llm_postprocess_endpoint_returns_run_summary(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path
    monkeypatch.setattr(main, "get_job", lambda job_id: {"id": job_id, "status": "done", "output_dir": str(output_dir)})
    monkeypatch.setattr(
        main,
        "run_qwen3_postprocess",
        lambda path: {
            "runId": "llm-test",
            "status": "done",
            "addedCount": 1,
            "skippedCount": 0,
            "auditDir": str(output_dir / "review" / "llm_runs" / "llm-test"),
            "addedCorrections": [],
            "skippedCorrections": [],
        },
    )
    monkeypatch.setattr(main, "ensure_inside_project", lambda path: Path(path))

    response = client.post("/api/jobs/job-1/llm/postprocess")

    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["runId"] == "llm-test"
    assert data["addedCount"] == 1


def test_llm_status_reports_loaded_model(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, settings):
            self.settings = settings

        def model_status(self) -> dict[str, Any]:
            return {
                "reachable": True,
                "modelLoaded": True,
                "configuredModel": self.settings.model,
                "availableModels": [self.settings.model],
                "currentModel": self.settings.model,
                "error": None,
            }

    monkeypatch.setattr(main, "OpenAICompatibleLLMClient", FakeClient)

    response = client.get("/api/llm/status")

    assert response.status_code == 200
    assert response.json()["modelLoaded"] is True


def test_llm_profile_put_returns_saved_profile(monkeypatch) -> None:
    class FakeProfile:
        def model_dump(self, mode: str) -> dict[str, Any]:
            return {"stage": "asr_correction", "profileId": "local-asr-correction"}

    monkeypatch.setattr(main, "save_local_profile", lambda stage, payload: FakeProfile())

    response = client.put("/api/llm/profiles/asr_correction", json={"profileId": "local-asr-correction"})

    assert response.status_code == 200
    assert response.json()["profile"]["profileId"] == "local-asr-correction"


def test_llm_run_endpoint_starts_async_run(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path
    monkeypatch.setattr(main, "get_job", lambda job_id: {"id": job_id, "status": "done", "output_dir": str(output_dir)})
    monkeypatch.setattr(main, "ensure_inside_project", lambda path: Path(path))
    monkeypatch.setattr(
        main,
        "start_llm_run",
        lambda **kwargs: {
            "run_id": "llm-test",
            "job_id": kwargs["job_id"],
            "stage": kwargs["request"].stage,
            "status": "queued",
            "events": [],
        },
    )

    response = client.post("/api/jobs/job-1/llm/runs", json={"stage": "asr_correction", "model": "qwen3-8b"})

    assert response.status_code == 200
    assert response.json()["run_id"] == "llm-test"


def test_get_llm_run_rejects_wrong_job(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "get_job", lambda job_id: {"id": job_id, "status": "done", "output_dir": str(tmp_path)})
    monkeypatch.setattr(main, "ensure_inside_project", lambda path: Path(path))
    monkeypatch.setattr(main, "get_llm_run", lambda run_id: {"run_id": run_id, "job_id": "other-job"})

    response = client.get("/api/jobs/job-1/llm/runs/llm-test")

    assert response.status_code == 404


def test_audio_endpoint_serves_only_original_upload(monkeypatch) -> None:
    tmp_root = PROJECT_ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        output_dir = Path(tmp)
        upload_path = UPLOADS_DIR / "api_audio_test.mp3"
        upload_path.write_bytes(b"fake-audio")
        try:
            (output_dir / "metadata.json").write_text(
                json.dumps({"input_file": str(upload_path)}, ensure_ascii=False),
                encoding="utf-8",
            )
            monkeypatch.setattr(main, "get_job", lambda job_id: {"id": job_id, "status": "done", "output_dir": str(output_dir)})

            response = client.get("/api/jobs/job-1/audio")

            assert response.status_code == 200
            assert response.content == b"fake-audio"
        finally:
            upload_path.unlink(missing_ok=True)


def test_audio_endpoint_rejects_non_upload_project_file(monkeypatch) -> None:
    tmp_root = PROJECT_ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        output_dir = Path(tmp)
        (output_dir / "metadata.json").write_text(
            json.dumps({"input_file": str(PROJECT_ROOT / "README.md")}, ensure_ascii=False),
            encoding="utf-8",
        )
        monkeypatch.setattr(main, "get_job", lambda job_id: {"id": job_id, "status": "done", "output_dir": str(output_dir)})

        response = client.get("/api/jobs/job-1/audio")

        assert response.status_code == 400
        assert "uploads" in response.json()["detail"]


def test_speaker_endpoint_updates_display_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "get_job", lambda job_id: {"id": job_id, "status": "done", "output_dir": str(tmp_path)})
    monkeypatch.setattr(main, "ensure_inside_project", lambda path: Path(path))
    monkeypatch.setattr(main, "update_speaker", lambda output_dir, label, payload: {"speaker": {"label": label, **payload}})

    response = client.patch("/api/jobs/job-1/speakers/SPEAKER_00", json={"displayName": "Ведущий"})

    assert response.status_code == 200
    assert response.json()["speaker"]["displayName"] == "Ведущий"


def test_segment_speaker_endpoint_requires_speaker(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "get_job", lambda job_id: {"id": job_id, "status": "done", "output_dir": str(tmp_path)})
    monkeypatch.setattr(main, "ensure_inside_project", lambda path: Path(path))

    response = client.patch("/api/jobs/job-1/segments/seg-000001/speaker", json={})

    assert response.status_code == 400
