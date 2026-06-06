from __future__ import annotations

from pathlib import Path

from app.env_utils import load_project_env


load_project_env()

PROJECT_ROOT = Path(r"Z:\Ogurcast")
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
ASR_MODELS_DIR = MODELS_DIR / "asr"
ALIGN_MODELS_DIR = MODELS_DIR / "align"
DIARIZATION_MODELS_DIR = MODELS_DIR / "diarization"
TMP_DIR = PROJECT_ROOT / "tmp"
CACHE_DIR = PROJECT_ROOT / ".cache"
STATIC_DIR = PROJECT_ROOT / "app" / "static"

DEFAULT_MODEL = "medium"
DEFAULT_LANGUAGE = "ru"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE_CUDA = "float16"
DEFAULT_COMPUTE_TYPE_CPU = "int8"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MIN_SPEAKERS = 2
DEFAULT_MAX_SPEAKERS = 2

ALLOWED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".webm",
    ".ogg",
    ".flac",
    ".aac",
    ".wma",
}


def ensure_project_dirs() -> None:
    for path in (
        UPLOADS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        MODELS_DIR,
        ASR_MODELS_DIR,
        ALIGN_MODELS_DIR,
        DIARIZATION_MODELS_DIR,
        TMP_DIR,
        CACHE_DIR,
        CACHE_DIR / "huggingface",
        CACHE_DIR / "huggingface" / "hub",
        CACHE_DIR / "huggingface" / "transformers",
        CACHE_DIR / "torch",
        CACHE_DIR / "pip",
        CACHE_DIR / "nltk",
    ):
        path.mkdir(parents=True, exist_ok=True)


ensure_project_dirs()
