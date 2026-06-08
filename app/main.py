from __future__ import annotations

import os
import platform
import shutil
import subprocess
import json
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
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
from app.file_utils import copy_upload_to_disk, ensure_inside_project, resolve_output_root, safe_filename, validate_audio_extension
from app.jobs import get_active_job, get_job, start_job
from app.llm_client import LLMSettings, OpenAICompatibleLLMClient
from app.llm_profiles import list_profiles, save_local_profile
from app.llm_postprocess import LLMPostprocessError, run_qwen3_postprocess
from app.llm_runs import LLMRunRequest, get_llm_run, start_llm_run
from app.review_artifacts import (
    apply_safe_asr_batch,
    ensure_review_artifacts,
    get_entities,
    get_transcript,
    get_words,
    review_artifact_paths,
    rollback_batch,
    reassign_segment_speaker,
    update_correction,
    update_entity,
    update_speaker,
)


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


@app.get("/api/llm/status")
def llm_status() -> dict[str, object]:
    settings = LLMSettings.from_env()
    client = OpenAICompatibleLLMClient(settings)
    status = client.model_status()
    return {
        "baseUrl": settings.base_url,
        "configuredModel": settings.model,
        "temperature": settings.temperature,
        "timeoutSec": settings.timeout_sec,
        "hasApiKey": bool(settings.api_key),
        **status,
    }


@app.get("/api/llm/profiles")
def llm_profiles() -> dict[str, object]:
    try:
        return list_profiles()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка LLM profiles: {exc}") from exc


@app.put("/api/llm/profiles/{stage}")
def put_llm_profile(stage: str, payload: dict[str, Any] | None = Body(default=None)) -> dict[str, object]:
    try:
        profile = save_local_profile(stage, payload or {})
        return {"profile": profile.model_dump(mode="json")}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Ошибка: LLM stage не найден.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


def _read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_job_output_dir(job_id: str) -> Path:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ошибка: задача не найдена.")
    if job["status"] != "done" or not job.get("output_dir"):
        raise HTTPException(status_code=400, detail="Review доступен только после завершения задачи.")

    try:
        output_dir = ensure_inside_project(Path(str(job["output_dir"])))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not output_dir.is_dir():
        raise HTTPException(status_code=400, detail="Ошибка: папка результата не найдена.")
    return output_dir


def _completed_job_audio_path(job_id: str) -> Path:
    output_dir = _completed_job_output_dir(job_id)
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        raise HTTPException(status_code=400, detail="Ошибка: metadata.json не найден.")
    try:
        metadata = _read_json_file(metadata_path)
        input_file = metadata.get("input_file") if isinstance(metadata, dict) else None
        if not isinstance(input_file, str) or not input_file.strip():
            raise ValueError("metadata.json не содержит input_file.")
        audio_path = ensure_inside_project(Path(input_file))
        uploads_root = UPLOADS_DIR.resolve(strict=False)
        try:
            audio_path.resolve(strict=False).relative_to(uploads_root)
        except ValueError as exc:
            raise ValueError("Исходное аудио должно находиться внутри uploads.") from exc
        validate_audio_extension(audio_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Ошибка: metadata.json поврежден.") from exc

    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Ошибка: исходный аудиофайл не найден.")
    return audio_path


def _review_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=400, detail=f"Ошибка: не найден артефакт {Path(str(exc)).name}.")
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="Ошибка: review-объект не найден.")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"Ошибка review: {exc}")


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str) -> FileResponse:
    audio_path = _completed_job_audio_path(job_id)
    media_type = mimetypes.guess_type(str(audio_path))[0] or "application/octet-stream"
    return FileResponse(audio_path, media_type=media_type, filename=audio_path.name)


@app.get("/api/jobs/{job_id}/artifacts")
def job_artifacts(job_id: str) -> dict[str, object]:
    output_dir = _completed_job_output_dir(job_id)
    try:
        ensure_review_artifacts(output_dir)
    except Exception as exc:
        raise _review_error(exc) from exc

    raw_files = [
        {"name": path.name, "path": str(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    ]
    return {
        "output_dir": str(output_dir),
        "raw_artifacts": raw_files,
        "review_artifacts": review_artifact_paths(output_dir),
    }


@app.get("/api/jobs/{job_id}/transcript")
def job_transcript(job_id: str) -> dict[str, object]:
    output_dir = _completed_job_output_dir(job_id)
    try:
        return get_transcript(output_dir)
    except Exception as exc:
        raise _review_error(exc) from exc


@app.get("/api/jobs/{job_id}/words")
def job_words(job_id: str) -> dict[str, object]:
    output_dir = _completed_job_output_dir(job_id)
    try:
        return {"words": get_words(output_dir)}
    except Exception as exc:
        raise _review_error(exc) from exc


@app.get("/api/jobs/{job_id}/review")
def job_review(job_id: str) -> dict[str, object]:
    output_dir = _completed_job_output_dir(job_id)
    try:
        return ensure_review_artifacts(output_dir)
    except Exception as exc:
        raise _review_error(exc) from exc


@app.patch("/api/jobs/{job_id}/corrections/{correction_id}")
def patch_correction(
    job_id: str,
    correction_id: str,
    payload: dict[str, Any] | None = Body(default=None),
) -> dict[str, object]:
    output_dir = _completed_job_output_dir(job_id)
    try:
        return update_correction(output_dir, correction_id, payload or {})
    except Exception as exc:
        raise _review_error(exc) from exc


@app.post("/api/jobs/{job_id}/corrections/apply-safe-asr")
def post_apply_safe_asr(job_id: str) -> dict[str, object]:
    output_dir = _completed_job_output_dir(job_id)
    try:
        return apply_safe_asr_batch(output_dir)
    except Exception as exc:
        raise _review_error(exc) from exc


@app.post("/api/jobs/{job_id}/corrections/rollback-batch")
def post_rollback_batch(
    job_id: str,
    payload: dict[str, Any] | None = Body(default=None),
) -> dict[str, object]:
    output_dir = _completed_job_output_dir(job_id)
    batch_id = payload.get("batchId") if payload else None
    try:
        return rollback_batch(output_dir, str(batch_id) if batch_id else None)
    except Exception as exc:
        raise _review_error(exc) from exc


@app.post("/api/jobs/{job_id}/llm/postprocess")
def post_llm_postprocess(job_id: str) -> dict[str, object]:
    output_dir = _completed_job_output_dir(job_id)
    try:
        return run_qwen3_postprocess(output_dir)
    except LLMPostprocessError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка LLM postprocessing ({exc.run_id}): {exc}",
        ) from exc
    except Exception as exc:
        raise _review_error(exc) from exc


@app.post("/api/jobs/{job_id}/llm/runs")
def post_llm_run(
    job_id: str,
    payload: dict[str, Any] | None = Body(default=None),
) -> dict[str, object]:
    output_dir = _completed_job_output_dir(job_id)
    try:
        request = LLMRunRequest.model_validate(payload or {})
        return start_llm_run(job_id=job_id, output_dir=output_dir, request=request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/llm/runs/{run_id}")
def get_job_llm_run(job_id: str, run_id: str) -> dict[str, object]:
    _completed_job_output_dir(job_id)
    run = get_llm_run(run_id)
    if run is None or run.get("job_id") != job_id:
        raise HTTPException(status_code=404, detail="Ошибка: LLM run не найден.")
    return run


@app.get("/api/jobs/{job_id}/entities")
def job_entities(job_id: str) -> dict[str, object]:
    output_dir = _completed_job_output_dir(job_id)
    try:
        return get_entities(output_dir)
    except Exception as exc:
        raise _review_error(exc) from exc


@app.patch("/api/jobs/{job_id}/entities/{entity_id}")
def patch_entity(
    job_id: str,
    entity_id: str,
    payload: dict[str, Any] | None = Body(default=None),
) -> dict[str, object]:
    output_dir = _completed_job_output_dir(job_id)
    try:
        return update_entity(output_dir, entity_id, payload or {})
    except Exception as exc:
        raise _review_error(exc) from exc


@app.patch("/api/jobs/{job_id}/speakers/{speaker_label}")
def patch_speaker(
    job_id: str,
    speaker_label: str,
    payload: dict[str, Any] | None = Body(default=None),
) -> dict[str, object]:
    output_dir = _completed_job_output_dir(job_id)
    try:
        return update_speaker(output_dir, speaker_label, payload or {})
    except Exception as exc:
        raise _review_error(exc) from exc


@app.patch("/api/jobs/{job_id}/segments/{segment_id}/speaker")
def patch_segment_speaker(
    job_id: str,
    segment_id: str,
    payload: dict[str, Any] | None = Body(default=None),
) -> dict[str, object]:
    output_dir = _completed_job_output_dir(job_id)
    speaker = (payload or {}).get("speaker")
    if not isinstance(speaker, str) or not speaker.strip():
        raise HTTPException(status_code=400, detail="speaker обязателен.")
    try:
        return reassign_segment_speaker(output_dir, segment_id, speaker.strip())
    except Exception as exc:
        raise _review_error(exc) from exc
