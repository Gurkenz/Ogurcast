from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

from app.config import ALLOWED_AUDIO_EXTENSIONS, OUTPUTS_DIR, PROJECT_ROOT


INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
RESERVED_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_filename(name: str) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub("_", name or "").strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "file"
    if cleaned.upper() in RESERVED_WINDOWS_NAMES:
        cleaned = f"{cleaned}_file"
    return cleaned[:180]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_audio_extension(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        raise ValueError(f"Ошибка: неподдерживаемый формат файла. Разрешены: {allowed}.")


def _is_inside_project(path: Path) -> bool:
    root = PROJECT_ROOT.resolve(strict=False)
    target = path.resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return True


def ensure_inside_project(path: Path) -> Path:
    if not _is_inside_project(path):
        raise ValueError(r"Папка вывода должна находиться внутри Z:\Ogurcast.")
    return path


def resolve_output_root(output_dir: str | Path | None) -> Path:
    raw_value = str(output_dir or "").strip()
    if not raw_value:
        candidate = OUTPUTS_DIR
    else:
        raw_path = Path(raw_value)
        candidate = raw_path if raw_path.is_absolute() else OUTPUTS_DIR / raw_path

    ensure_inside_project(candidate)
    return ensure_dir(candidate)


def create_run_dir(output_root: Path, input_stem: str) -> Path:
    ensure_inside_project(output_root)
    safe_stem = safe_filename(input_stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(output_root / f"{safe_stem}_whisperx_{timestamp}")


def copy_upload_to_disk(upload_file, dest_path: Path) -> None:
    ensure_dir(dest_path.parent)
    upload_file.file.seek(0)
    with dest_path.open("wb") as dest:
        shutil.copyfileobj(upload_file.file, dest)
