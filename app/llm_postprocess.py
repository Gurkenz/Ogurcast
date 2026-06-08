from __future__ import annotations

from pathlib import Path
from typing import Any

from app.llm_runs import LLMClientProtocol, LLMPostprocessError, LLMRunRequest, run_llm_stage


def run_qwen3_postprocess(output_dir: Path, client: LLMClientProtocol | None = None) -> dict[str, Any]:
    return run_llm_stage(
        output_dir,
        request=LLMRunRequest(stage="asr_correction"),
        client=client,
    )
