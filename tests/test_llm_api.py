from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

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
