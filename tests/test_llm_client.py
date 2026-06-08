from __future__ import annotations

from typing import Any

from app.llm_client import LLMSettings, OpenAICompatibleLLMClient


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError("fallback response should be successful")


class FakeHttpClient:
    post_payloads: list[dict[str, Any]] = []

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def __enter__(self) -> "FakeHttpClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> FakeResponse:
        self.post_payloads.append(json)
        if len(self.post_payloads) == 1:
            return FakeResponse(400, {"error": "response_format unsupported"})
        return FakeResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"version":1,"model":"qwen3-8b","corrections":[]}',
                        }
                    }
                ]
            },
        )


def test_complete_json_retries_without_response_format(monkeypatch) -> None:
    import app.llm_client as llm_client

    FakeHttpClient.post_payloads = []
    monkeypatch.setattr(llm_client.httpx, "Client", FakeHttpClient)
    client = OpenAICompatibleLLMClient(
        LLMSettings(
            base_url="http://127.0.0.1:1234/v1",
            model="qwen3-8b",
            temperature=0.1,
            timeout_sec=120,
        )
    )

    result = client.complete_json([{"role": "user", "content": "test"}])

    assert result["content"]["corrections"] == []
    assert "response_format" in FakeHttpClient.post_payloads[0]
    assert "response_format" not in FakeHttpClient.post_payloads[1]
