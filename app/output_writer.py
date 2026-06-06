from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _format_timestamp(seconds: Any, separator: str = ".") -> str:
    total_ms = max(0, int(round(_to_float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _segment_speaker(segment: dict[str, Any]) -> str:
    speaker = segment.get("speaker")
    if speaker:
        return str(speaker)

    speakers = [
        str(word["speaker"])
        for word in segment.get("words", [])
        if isinstance(word, dict) and word.get("speaker")
    ]
    if speakers:
        return Counter(speakers).most_common(1)[0][0]
    return "UNKNOWN"


def _word_source(result: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(result.get("word_segments"), list):
        return [word for word in result["word_segments"] if isinstance(word, dict)]

    words: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        if not isinstance(segment, dict):
            continue
        for word in segment.get("words", []):
            if isinstance(word, dict):
                words.append(word)
    return words


def _flatten_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    flattened = []
    for word in _word_source(result):
        flattened.append(
            {
                "start": _to_float(word.get("start")),
                "end": _to_float(word.get("end")),
                "word": str(word.get("word") or "").strip(),
                "speaker": word.get("speaker") or "UNKNOWN",
                "score": word.get("score"),
            }
        )
    return flattened


def _plain_transcript(segments: list[dict[str, Any]]) -> str:
    lines = [str(segment.get("text") or "").strip() for segment in segments]
    return "\n".join(line for line in lines if line).strip() + "\n"


def _speaker_transcript(segments: list[dict[str, Any]]) -> str:
    blocks: list[dict[str, Any]] = []
    for segment in segments:
        speaker = _segment_speaker(segment)
        text = str(segment.get("text") or "").strip()
        if not text:
            continue

        start = _to_float(segment.get("start"))
        end = _to_float(segment.get("end"), start)
        if blocks and blocks[-1]["speaker"] == speaker:
            blocks[-1]["end"] = end
            blocks[-1]["texts"].append(text)
        else:
            blocks.append({"speaker": speaker, "start": start, "end": end, "texts": [text]})

    rendered = []
    for block in blocks:
        rendered.append(
            f"[{_format_timestamp(block['start'])} - {_format_timestamp(block['end'])}] {block['speaker']}:\n"
            f"{' '.join(block['texts'])}"
        )
    return "\n\n".join(rendered).strip() + "\n"


def _srt(segments: list[dict[str, Any]]) -> str:
    entries = []
    for index, segment in enumerate(segments, start=1):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = _format_timestamp(segment.get("start"), separator=",")
        end = _format_timestamp(segment.get("end"), separator=",")
        entries.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(entries).strip() + "\n"


def _vtt(segments: list[dict[str, Any]]) -> str:
    entries = ["WEBVTT", ""]
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = _format_timestamp(segment.get("start"))
        end = _format_timestamp(segment.get("end"))
        entries.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(entries).strip() + "\n"


def write_outputs(
    result: dict[str, Any],
    run_dir: Path,
    metadata: dict[str, Any],
    log_lines: list[str] | None = None,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    segments = [segment for segment in result.get("segments", []) if isinstance(segment, dict)]

    files = {
        "result_raw": run_dir / "result_raw.json",
        "segments": run_dir / "segments.json",
        "words": run_dir / "words.json",
        "transcript": run_dir / "transcript.txt",
        "speaker_transcript": run_dir / "speaker_transcript.txt",
        "srt": run_dir / "transcript.srt",
        "vtt": run_dir / "transcript.vtt",
        "metadata": run_dir / "metadata.json",
        "log": run_dir / "run.log",
    }

    _write_json(files["result_raw"], result)
    _write_json(files["segments"], segments)
    _write_json(files["words"], _flatten_words(result))
    files["transcript"].write_text(_plain_transcript(segments), encoding="utf-8")
    files["speaker_transcript"].write_text(_speaker_transcript(segments), encoding="utf-8")
    files["srt"].write_text(_srt(segments), encoding="utf-8")
    files["vtt"].write_text(_vtt(segments), encoding="utf-8")
    _write_json(files["metadata"], metadata)

    if log_lines is not None:
        files["log"].write_text("\n".join(log_lines).strip() + "\n", encoding="utf-8")
    elif not files["log"].exists():
        files["log"].write_text("", encoding="utf-8")

    return {name: str(path) for name, path in files.items()}
