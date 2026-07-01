from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from app.llm_runs import start_auto_llm_audit
from app.review_artifacts import add_technical_flag, ensure_review_artifacts
from app.whisperx_runner import run_whisperx_job


JobStatus = Literal["queued", "running", "done", "error"]


@dataclass
class JobState:
    id: str
    status: JobStatus
    progress: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: str | None = None
    finished_at: str | None = None
    output_dir: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


_lock = threading.RLock()
_jobs: dict[str, JobState] = {}
_active_job_id: str | None = None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_progress(job_id: str, message: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.progress.append(message)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        return asdict(job) if job is not None else None


def get_active_job() -> dict[str, Any] | None:
    with _lock:
        if _active_job_id is None:
            return None
        job = _jobs.get(_active_job_id)
        if job is None or job.status not in {"queued", "running"}:
            return None
        return asdict(job)


def start_job(
    *,
    input_path: Path,
    output_root: Path,
    model_name: str,
    language: str,
    device: str,
    compute_type: str,
    batch_size: int,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
    hf_token: str | None,
) -> dict[str, Any]:
    global _active_job_id

    with _lock:
        active = get_active_job()
        if active is not None:
            raise RuntimeError("Уже выполняется другая задача. Дождитесь завершения.")

        job_id = str(uuid.uuid4())
        job = JobState(id=job_id, status="queued")
        job.progress.append("Задача поставлена в очередь.")
        _jobs[job_id] = job
        _active_job_id = job_id

    thread = threading.Thread(
        target=_run_job_thread,
        kwargs={
            "job_id": job_id,
            "input_path": input_path,
            "output_root": output_root,
            "model_name": model_name,
            "language": language,
            "device": device,
            "compute_type": compute_type,
            "batch_size": batch_size,
            "diarize": diarize,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "hf_token": hf_token,
        },
        daemon=True,
    )
    thread.start()
    return asdict(job)


def _run_job_thread(
    *,
    job_id: str,
    input_path: Path,
    output_root: Path,
    model_name: str,
    language: str,
    device: str,
    compute_type: str,
    batch_size: int,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
    hf_token: str | None,
) -> None:
    global _active_job_id

    with _lock:
        job = _jobs[job_id]
        job.status = "running"
        job.started_at = _now()
        job.progress.append("Задача запущена.")

    try:
        result = run_whisperx_job(
            input_path=input_path,
            output_root=output_root,
            model_name=model_name,
            language=language,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=hf_token,
            progress_callback=lambda message: append_progress(job_id, message),
        )
        with _lock:
            job = _jobs[job_id]
            job.status = "done"
            job.finished_at = _now()
            job.output_dir = result.get("output_dir")
            job.result = result
            job.progress.append("Готово.")
        _after_successful_job(job_id, result)
    except Exception as exc:
        with _lock:
            job = _jobs[job_id]
            job.status = "error"
            job.finished_at = _now()
            job.error = str(exc)
            job.progress.append(str(exc))
    finally:
        with _lock:
            if _active_job_id == job_id:
                _active_job_id = None


def _after_successful_job(job_id: str, result: dict[str, Any]) -> None:
    output_dir_value = result.get("output_dir")
    if not output_dir_value:
        return
    output_dir = Path(str(output_dir_value))
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    diarization_status = str(metadata.get("diarization_status") or "")
    try:
        ensure_review_artifacts(output_dir)
        if diarization_status == "done":
            run = start_auto_llm_audit(job_id=job_id, output_dir=output_dir)
            _append_auto_audit_progress(job_id, run)
        else:
            add_technical_flag(
                output_dir,
                flag="diarization_not_done",
                severity="high" if diarization_status == "failed" else "medium",
                risk="high",
                reason=f"LLM audit не запущен: diarization_status={diarization_status or 'unknown'}.",
                recommended_action="Проверить HF_TOKEN/pyannote и повторить диаризацию перед автоматическим LLM audit.",
            )
            append_progress(job_id, "LLM audit пропущен: диаризация не завершена.")
    except Exception as exc:
        append_progress(job_id, f"LLM audit не запущен: {exc}")


def _append_auto_audit_progress(job_id: str, run: dict[str, Any]) -> None:
    status = str(run.get("status") or "")
    if status == "deduplicated":
        append_progress(job_id, "LLM audit уже есть для этих артефактов.")
    elif status == "disabled":
        append_progress(job_id, "LLM audit отключен настройкой OGURCAST_LLM_AUTO_AUDIT.")
    else:
        append_progress(job_id, "LLM audit поставлен в очередь.")
