#!/usr/bin/env python3
"""Download official GigaAM-v3 and pyannote assets into the Ogurcast project.

Run from Z:\\Ogurcast (Python 3.11):
    .\\.venv\\Scripts\\python.exe .\\scripts\\download_gigaam_assets.py

This downloads model files only. It does not install or update torch/WhisperX.
Use --skip-pyannote if Hugging Face model agreements are not yet accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ASR_DIR = ROOT / "models" / "asr" / "gigaam"
DIAR_DIR = ROOT / "models" / "diarization"
HUB_DIR = ROOT / ".cache" / "huggingface" / "hub"
TMP_DIR = ROOT / "tmp"
BASE_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"

# MD5 values copied from salute-developers/GigaAM/gigaam/__init__.py.
GIGAAM = (
    ("v3_rnnt.ckpt", "0fd2c9a1ff66abd8d32a3a07f7592815"),
    ("v3_e2e_rnnt.ckpt", "2730de7545ac43ad256485a462b0a27a"),
    ("v3_e2e_rnnt_tokenizer.model", None),
)

# Both models require accepting the agreements in the user's HF account.
PYANNOTE = (
    "pyannote/speaker-diarization-community-1",
    "pyannote/segmentation-3.0",
)


def file_hash(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def model_info(path: Path, url: str, expected_md5: str | None = None) -> dict:
    return {
        "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "source": url,
        "bytes": path.stat().st_size,
        "md5": file_hash(path, "md5"),
        "sha256": file_hash(path, "sha256"),
        "official_md5": expected_md5,
    }


def download_model(name: str, expected_md5: str | None) -> dict:
    ASR_DIR.mkdir(parents=True, exist_ok=True)
    dest = ASR_DIR / name
    url = f"{BASE_URL}/{name}"
    if dest.is_file() and dest.stat().st_size > 1024:
        if expected_md5 is None or file_hash(dest, "md5") == expected_md5:
            print(f"Verified existing: {dest}")
            return model_info(dest, url, expected_md5)
        print(f"Invalid checksum; will replace: {dest}")

    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, 4):
        try:
            if part.exists():
                part.unlink()
            request = Request(url, headers={"User-Agent": "Ogurcast-Model-Downloader/1.0"})
            with urlopen(request, timeout=90) as response, part.open("wb") as output:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type.lower():
                    raise RuntimeError(f"Unexpected HTML response for {name}")
                copied = 0
                while True:
                    data = response.read(4 * 1024 * 1024)
                    if not data:
                        break
                    output.write(data)
                    copied += len(data)
                    if copied // (128 * 1024 * 1024) != (copied - len(data)) // (128 * 1024 * 1024):
                        print(f"  {name}: {copied // (1024 * 1024)} MiB downloaded")
            if part.stat().st_size <= 1024:
                raise RuntimeError(f"Unexpectedly small file: {name}")
            if expected_md5 is not None:
                actual_md5 = file_hash(part, "md5")
                if actual_md5 != expected_md5:
                    raise RuntimeError(
                        f"MD5 mismatch for {name}: expected {expected_md5}, received {actual_md5}"
                    )
            part.replace(dest)
            return model_info(dest, url, expected_md5)
        except Exception as exc:
            print(f"  Download failed ({attempt}/3): {type(exc).__name__}: {exc}", file=sys.stderr)
            if part.exists():
                part.unlink()
            if attempt == 3:
                raise
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {name}")


def token_from_env() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if stripped.startswith("HF_TOKEN="):
                return stripped.partition("=")[2].strip().strip("\\\"'") or None
    return None


def download_pyannote() -> dict:
    token = token_from_env()
    if not token:
        raise RuntimeError(
            "Missing HF_TOKEN. Add it to Z:\\Ogurcast\\.env or the process environment. "
            "Accept both model agreements on Hugging Face first."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed in this Python environment."
        ) from exc

    DIAR_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for repo in PYANNOTE:
        print(f"Downloading gated model {repo}...")
        cached = Path(snapshot_download(
            repo_id=repo,
            token=token,
            cache_dir=str(HUB_DIR),
        ))
        target = DIAR_DIR / repo.replace("/", "--")
        shutil.copytree(cached, target, dirs_exist_ok=True)
        files = [
            {"path": str(p.relative_to(target)).replace("\\", "/"),
             "bytes": p.stat().st_size, "sha256": file_hash(p, "sha256")}
            for p in sorted(target.rglob("*")) if p.is_file()
        ]
        manifest[repo] = {
            "url": f"https://huggingface.co/{repo}",
            "local_path": str(target.relative_to(ROOT)).replace("\\", "/"),
            "files": files,
        }
        print(f"Saved: {target}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download GigaAM weights for Ogurcast")
    parser.add_argument("--skip-pyannote", action="store_true",
                        help="Download only public GigaAM weights")
    args = parser.parse_args()

    for directory in (ASR_DIR, DIAR_DIR, HUB_DIR, TMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # Keep Hugging Face downloads and temporary files inside this project.
    os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(HUB_DIR)
    os.environ["HF_TOKEN_PATH"] = str(ROOT / ".cache" / "huggingface" / "token")
    os.environ["TMP"] = str(TMP_DIR)
    os.environ["TEMP"] = str(TMP_DIR)

    report = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(ROOT),
        "models": {},
        "pyannote": {},
        "errors": [],
    }

    for name, checksum in GIGAAM:
        print(f"Downloading/verifying: {name}")
        try:
            report["models"][name] = download_model(name, checksum)
        except Exception as exc:
            report["errors"].append(f"{name}: {type(exc).__name__}: {exc}")

    if not args.skip_pyannote:
        try:
            report["pyannote"] = download_pyannote()
        except Exception as exc:
            report["errors"].append(f"pyannote: {type(exc).__name__}: {exc}")
            print(
                "Pyannote access requires agreements at:\n"
                "https://huggingface.co/pyannote/speaker-diarization-community-1\n"
                "https://huggingface.co/pyannote/segmentation-3.0",
                file=sys.stderr,
            )

    manifest_path = ROOT / "models" / "gigaam_assets_manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    if report["errors"]:
        for issue in report["errors"]:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1
    print("Download and verification completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
