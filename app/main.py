from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COMPUTE_TYPE_CUDA,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_SPEAKERS,
    DEFAULT_MIN_SPEAKERS,
    DEFAULT_MODEL,
    PROJECT_ROOT,
    STATIC_DIR,
    UPLOADS_DIR,
)
from app.env_utils import load_project_env
from app.file_utils import copy_upload_to_disk, resolve_output_root, safe_filename, validate_audio_extension
from app.jobs import get_active_job, get_job, start_job


load_project_env()

app = FastAPI(title="Ogurcast WhisperX Test")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _torch_health() -> tuple[str, bool, str]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if cuda_available else "NO CUDA"
        return torch.__version__, cuda_available, gpu
    except Exception:
        return "not-installed", False, "NO CUDA"


def _whisperx_ok() -> bool:
    try:
        import whisperx  # noqa: F401
    except Exception:
        return False
    return True


def _ffmpeg_ok() -> bool:
    candidates = [os.getenv("OGURCAST_FFMPEG"), shutil.which("ffmpeg.exe"), shutil.which("ffmpeg"), "ffmpeg"]
    checked: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.add(candidate)
        try:
            subprocess.run([candidate, "-version"], capture_output=True, check=True)
            return True
        except (OSError, subprocess.CalledProcessError):
            continue
    return False


@app.get("/api/health")
def health() -> dict[str, object]:
    torch_version, cuda_available, gpu = _torch_health()
    return {
        "project": "Ogurcast",
        "project_root": str(PROJECT_ROOT),
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda_available": cuda_available,
        "gpu": gpu,
        "whisperx_ok": _whisperx_ok(),
        "ffmpeg_ok": _ffmpeg_ok(),
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
        "hf_home": os.getenv("HF_HOME"),
        "torch_home": os.getenv("TORCH_HOME"),
        "tmp": os.getenv("TMP"),
    }


@app.post("/api/jobs")
def create_job(
    file: UploadFile = File(...),
    output_dir: str = Form(str(PROJECT_ROOT / "outputs")),
    model: str = Form(DEFAULT_MODEL),
    language: str = Form(DEFAULT_LANGUAGE),
    device: str = Form(DEFAULT_DEVICE),
    compute_type: str = Form(DEFAULT_COMPUTE_TYPE_CUDA),
    batch_size: int = Form(DEFAULT_BATCH_SIZE),
    diarize: bool = Form(True),
    min_speakers: int | None = Form(DEFAULT_MIN_SPEAKERS),
    max_speakers: int | None = Form(DEFAULT_MAX_SPEAKERS),
    hf_token: str | None = Form(None),
) -> dict[str, object]:
    if get_active_job() is not None:
        raise HTTPException(status_code=409, detail="Уже выполняется другая задача. Дождитесь завершения.")

    filename = file.filename or "upload"
    try:
        validate_audio_extension(Path(filename))
        output_root = resolve_output_root(output_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    safe_name = safe_filename(filename)
    upload_path = UPLOADS_DIR / safe_name
    if upload_path.exists():
        stem = Path(safe_name).stem
        suffix = Path(safe_name).suffix
        upload_path = UPLOADS_DIR / f"{stem}_{os.getpid()}{suffix}"

    try:
        copy_upload_to_disk(file, upload_path)
        job = start_job(
            input_path=upload_path,
            output_root=output_root,
            model_name=model,
            language=language,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=hf_token.strip() if hf_token and hf_token.strip() else None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка: {exc}") from exc

    return {"job_id": job["id"], "status": job["status"], "message": "Задача запущена."}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, object]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ошибка: задача не найдена.")
    return job


@app.get("/api/jobs/{job_id}/files")
def job_files(job_id: str) -> dict[str, object]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ошибка: задача не найдена.")
    if job["status"] != "done" or not job.get("output_dir"):
        raise HTTPException(status_code=400, detail="Файлы доступны только после завершения задачи.")

    output_dir = Path(str(job["output_dir"]))
    files = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            files.append({"name": path.name, "path": str(path)})
    return {"output_dir": str(output_dir), "files": files}
