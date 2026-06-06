from __future__ import annotations

import argparse
import gc
import inspect
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable

from app.config import (
    ALIGN_MODELS_DIR,
    ASR_MODELS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_COMPUTE_TYPE_CUDA,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_SPEAKERS,
    DEFAULT_MIN_SPEAKERS,
    DEFAULT_MODEL,
    LOGS_DIR,
    PROJECT_ROOT,
)
from app.env_utils import load_project_env
from app.file_utils import create_run_dir, resolve_output_root, validate_audio_extension
from app.output_writer import write_outputs


load_project_env()

CUDA_OOM_MESSAGE = (
    "Недостаточно видеопамяти. Уменьшите batch_size до 4, поставьте "
    "compute_type=int8 или выберите модель smaller."
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sanitize_secret(text: str) -> str:
    token = os.getenv("HF_TOKEN")
    if token:
        text = text.replace(token, "***")
    return text


def _error_message(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "Ошибка: файл не найден."

    raw = _sanitize_secret(str(exc))
    lower = raw.lower()
    if "out of memory" in lower and "cuda" in lower:
        return CUDA_OOM_MESSAGE
    if "cuda" in lower and ("not available" in lower or "недоступ" in lower):
        return "Ошибка: CUDA недоступна."
    if "ffmpeg" in lower or "file contains data in an unknown format" in lower:
        return "Ошибка: не найден FFmpeg или FFmpeg не смог прочитать файл."
    if "hf_token" in lower or "auth token" in lower or "access token" in lower:
        return "Ошибка: отсутствует HF_TOKEN."
    if "401" in lower or "403" in lower or "gated" in lower or "pyannote" in lower:
        return "Ошибка: нет доступа к модели pyannote. Проверьте HF_TOKEN и принятие model agreement."
    if raw.startswith("Ошибка") or raw.startswith("Папка вывода"):
        return raw
    return f"Ошибка WhisperX: {raw}"


def _resolve_ffmpeg_executable() -> str:
    candidates = [
        os.getenv("OGURCAST_FFMPEG"),
        shutil.which("ffmpeg.exe"),
        shutil.which("ffmpeg"),
        "ffmpeg",
    ]
    checked: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.add(candidate)
        try:
            subprocess.run([candidate, "-version"], capture_output=True, check=True)
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("Ошибка: не найден FFmpeg.")


def _load_audio_with_ffmpeg(whisperx_module: Any, input_path: Path, ffmpeg_path: str):
    import numpy as np

    sample_rate = getattr(getattr(whisperx_module, "audio", None), "SAMPLE_RATE", 16000)
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(input_path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to load audio: {stderr}") from exc
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Ошибка: не установлен torch. Запустите scripts\\install_deps.ps1.") from exc
    return torch


def _import_whisperx():
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError("Ошибка: не установлен whisperx. Запустите scripts\\install_deps.ps1.") from exc
    return whisperx


def _check_device(torch_module, device: str) -> None:
    if device == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("Ошибка: CUDA недоступна.")


def _load_asr_model(
    whisperx_module: Any,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
) -> Any:
    kwargs: dict[str, Any] = {
        "compute_type": compute_type,
        "download_root": str(ASR_MODELS_DIR),
        "language": language,
    }
    try:
        return whisperx_module.load_model(model_name, device, **kwargs)
    except TypeError as exc:
        if "language" not in str(exc):
            raise
        kwargs.pop("language", None)
        return whisperx_module.load_model(model_name, device, **kwargs)


def _call_transcribe(model: Any, audio: Any, batch_size: int, language: str) -> dict[str, Any]:
    try:
        return model.transcribe(audio, batch_size=batch_size, language=language)
    except TypeError as exc:
        if "language" not in str(exc):
            raise
        return model.transcribe(audio, batch_size=batch_size)


def _call_align(whisperx_module: Any, segments: list[dict[str, Any]], model_a: Any, metadata: Any, audio: Any, device: str):
    try:
        return whisperx_module.align(
            segments,
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except TypeError as exc:
        if "return_char_alignments" not in str(exc):
            raise
        return whisperx_module.align(segments, model_a, metadata, audio, device)


def _load_align_model(whisperx_module: Any, language_code: str, device: str) -> tuple[Any, Any]:
    kwargs: dict[str, Any] = {
        "language_code": language_code,
        "device": device,
        "model_dir": str(ALIGN_MODELS_DIR),
    }
    try:
        return whisperx_module.load_align_model(**kwargs)
    except TypeError as exc:
        if "model_dir" not in str(exc):
            raise
        kwargs.pop("model_dir", None)
        return whisperx_module.load_align_model(**kwargs)


def _load_diarization_pipeline(device: str, token: str):
    from whisperx.diarize import DiarizationPipeline

    kwargs: dict[str, Any] = {"device": device}
    try:
        parameters = inspect.signature(DiarizationPipeline).parameters
        if "use_auth_token" in parameters:
            kwargs["use_auth_token"] = token
        elif "token" in parameters:
            kwargs["token"] = token
        else:
            kwargs["use_auth_token"] = token
    except (TypeError, ValueError):
        kwargs["use_auth_token"] = token
    return DiarizationPipeline(**kwargs)


def _call_diarization(diarize_model: Any, audio: Any, min_speakers: int | None, max_speakers: int | None):
    kwargs: dict[str, Any] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    return diarize_model(audio, **kwargs)


def _version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def _versions() -> dict[str, str]:
    torch_version = "not-installed"
    try:
        import torch

        torch_version = torch.__version__
    except Exception:
        pass

    return {
        "python": platform.python_version(),
        "torch": torch_version,
        "whisperx": _version("whisperx"),
    }


def _metadata(
    *,
    input_path: Path,
    run_dir: Path,
    model_name: str,
    language: str,
    device: str,
    compute_type: str,
    batch_size: int,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
    alignment_status: str,
    diarization_status: str,
    started_at: str,
    finished_at: str,
    elapsed_sec: float,
) -> dict[str, Any]:
    return {
        "project": "Ogurcast",
        "project_root": str(PROJECT_ROOT),
        "input_file": str(input_path),
        "output_dir": str(run_dir),
        "model": model_name,
        "language": language,
        "device": device,
        "compute_type": compute_type,
        "batch_size": batch_size,
        "diarize": diarize,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "alignment_status": alignment_status,
        "diarization_status": diarization_status,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_sec": round(elapsed_sec, 3),
        "versions": _versions(),
    }


def _free_cuda(torch_module: Any) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def run_whisperx_job(
    input_path: Path,
    output_root: Path,
    model_name: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE_CUDA,
    batch_size: int = DEFAULT_BATCH_SIZE,
    diarize: bool = True,
    min_speakers: int | None = DEFAULT_MIN_SPEAKERS,
    max_speakers: int | None = DEFAULT_MAX_SPEAKERS,
    hf_token: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    load_project_env()

    input_path = Path(input_path)
    output_root = resolve_output_root(output_root)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    validate_audio_extension(input_path)
    ffmpeg_path = _resolve_ffmpeg_executable()

    run_dir = create_run_dir(output_root, input_path.stem)
    log_path = run_dir / "run.log"
    log_lines: list[str] = []

    def log(message: str) -> None:
        safe_message = _sanitize_secret(message)
        line = f"{_now()} {safe_message}"
        log_lines.append(line)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        if progress_callback is not None:
            progress_callback(safe_message)
        print(safe_message, flush=True)

    started_at = _now()
    started = time.monotonic()
    alignment_status = "skipped"
    diarization_status = "disabled" if not diarize else "failed"
    result: dict[str, Any] = {"segments": []}

    try:
        torch = _import_torch()
        whisperx = _import_whisperx()
        _check_device(torch, device)

        log("Загрузка аудио...")
        audio = _load_audio_with_ffmpeg(whisperx, input_path, ffmpeg_path)

        log("Загрузка ASR-модели...")
        model = _load_asr_model(whisperx, model_name, device, compute_type, language)
        log("Идет распознавание речи.")
        result = _call_transcribe(model, audio, batch_size=batch_size, language=language)
        log("Распознавание речи завершено.")
        del model
        _free_cuda(torch)

        try:
            log("Идет выравнивание таймкодов.")
            align_language = result.get("language") or language
            model_a, align_metadata = _load_align_model(whisperx, align_language, device)
            result = _call_align(whisperx, result["segments"], model_a, align_metadata, audio, device)
            alignment_status = "done"
            log("Выравнивание таймкодов завершено.")
            del model_a
            _free_cuda(torch)
        except Exception as exc:
            alignment_status = "failed"
            log(f"Ошибка выравнивания: {_error_message(exc)} Продолжаю с ASR-результатом.")

        if diarize:
            token = hf_token or os.getenv("HF_TOKEN")
            if not token:
                diarization_status = "failed"
                log("Ошибка: отсутствует HF_TOKEN. Диаризация пропущена.")
            else:
                try:
                    log("Идет диаризация.")
                    diarize_model = _load_diarization_pipeline(device, token)
                    diarize_segments = _call_diarization(diarize_model, audio, min_speakers, max_speakers)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    diarization_status = "done"
                    log("Диаризация завершена.")
                    del diarize_model
                    _free_cuda(torch)
                except Exception as exc:
                    diarization_status = "failed"
                    log(f"Ошибка диаризации: {_error_message(exc)} Продолжаю без спикеров.")

        finished_at = _now()
        metadata = _metadata(
            input_path=input_path,
            run_dir=run_dir,
            model_name=model_name,
            language=language,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            alignment_status=alignment_status,
            diarization_status=diarization_status,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_sec=time.monotonic() - started,
        )
        files = write_outputs(result, run_dir, metadata, log_lines)
        log("Готово.")
        return {"output_dir": str(run_dir), "files": files, "metadata": metadata}
    except Exception as exc:
        message = _error_message(exc)
        log(f"Ошибка: {message}" if not message.startswith("Ошибка") else message)
        raise RuntimeError(message) from exc


def warmup_models(
    model_name: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE_CUDA,
) -> None:
    load_project_env()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "warmup.log"
    lines: list[str] = []

    def log(message: str) -> None:
        line = f"{_now()} {message}"
        lines.append(line)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(message, flush=True)

    torch = _import_torch()
    whisperx = _import_whisperx()
    _check_device(torch, device)

    log("Загрузка ASR-модели...")
    model = _load_asr_model(whisperx, model_name, device, compute_type, language)
    log("ASR-модель загружена.")
    del model
    _free_cuda(torch)

    log("Загрузка alignment-модели...")
    model_a, _ = _load_align_model(whisperx, language, device)
    log("Alignment-модель загружена.")
    del model_a
    _free_cuda(torch)

    if os.getenv("HF_TOKEN"):
        log("Проверка diarization pipeline...")
        diarize_model = _load_diarization_pipeline(device, os.environ["HF_TOKEN"])
        log("Diarization pipeline инициализирован.")
        del diarize_model
        _free_cuda(torch)
    else:
        log("HF_TOKEN отсутствует. Diarization pipeline пропущен.")

    log("Warmup завершен.")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Локальный запуск Ogurcast WhisperX.")
    parser.add_argument("--warmup", action="store_true", help="Загрузить модели без обработки файла.")
    parser.add_argument("--input", type=Path, help="Путь к аудио или видео файлу.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs", help="Папка вывода.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Модель WhisperX.")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Язык аудио.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cuda", "cpu"], help="Устройство.")
    parser.add_argument("--compute-type", default=DEFAULT_COMPUTE_TYPE_CUDA, help="Compute type.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--diarize", action="store_true", help="Включить диаризацию.")
    parser.add_argument("--min-speakers", type=int, default=DEFAULT_MIN_SPEAKERS, help="Минимум спикеров.")
    parser.add_argument("--max-speakers", type=int, default=DEFAULT_MAX_SPEAKERS, help="Максимум спикеров.")
    parser.add_argument("--hf-token", default=None, help="HF token для текущего запуска. Значение не логируется.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        if args.warmup:
            warmup_models(
                model_name=args.model,
                language=args.language,
                device=args.device,
                compute_type=args.compute_type,
            )
            return 0

        if args.input is None:
            print("Ошибка: укажите --input или --warmup.", file=sys.stderr)
            return 2

        run_whisperx_job(
            input_path=args.input,
            output_root=args.output_dir,
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            diarize=args.diarize,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            hf_token=args.hf_token,
        )
        return 0
    except Exception as exc:
        print(_error_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
