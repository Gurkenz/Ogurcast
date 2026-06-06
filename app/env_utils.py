from __future__ import annotations

import os
import shutil
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - used before dependencies are installed
    load_dotenv = None


PROJECT_ROOT = Path(r"Z:\Ogurcast")


def _load_env_without_dependency(env_path: Path) -> None:
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


def load_project_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        if load_dotenv is not None:
            load_dotenv(env_path, override=False)
        else:
            _load_env_without_dependency(env_path)

    defaults = {
        "OGURCAST_ROOT": str(PROJECT_ROOT),
        "HF_HOME": str(PROJECT_ROOT / ".cache" / "huggingface"),
        "HF_HUB_CACHE": str(PROJECT_ROOT / ".cache" / "huggingface" / "hub"),
        "HF_TOKEN_PATH": str(PROJECT_ROOT / ".cache" / "huggingface" / "token"),
        "TRANSFORMERS_CACHE": str(PROJECT_ROOT / ".cache" / "huggingface" / "transformers"),
        "TORCH_HOME": str(PROJECT_ROOT / ".cache" / "torch"),
        "PIP_CACHE_DIR": str(PROJECT_ROOT / ".cache" / "pip"),
        "NLTK_DATA": str(PROJECT_ROOT / ".cache" / "nltk"),
        "TMP": str(PROJECT_ROOT / "tmp"),
        "TEMP": str(PROJECT_ROOT / "tmp"),
    }

    # Runtime processes often inherit TMP/TEMP/PATH-related values from Windows.
    # Project-local paths are not optional here: external caches must not fall
    # back to AppData or system temp just because a parent process had them set.
    for key, value in defaults.items():
        if key == "OGURCAST_ROOT":
            os.environ.setdefault(key, value)
        else:
            os.environ[key] = value

    tools_dir = PROJECT_ROOT / "tools"
    if not os.environ.get("OGURCAST_FFMPEG"):
        ffmpeg_path = shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")
        if ffmpeg_path and Path(ffmpeg_path).parent.resolve(strict=False) != tools_dir.resolve(strict=False):
            os.environ["OGURCAST_FFMPEG"] = ffmpeg_path

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    tools_dir_text = str(tools_dir)
    if tools_dir_text not in path_parts:
        os.environ["PATH"] = os.pathsep.join([tools_dir_text, *path_parts])

    for key, value in defaults.items():
        path = Path(value)
        if key == "HF_TOKEN_PATH":
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    tools_dir.mkdir(parents=True, exist_ok=True)
