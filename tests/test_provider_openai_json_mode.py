import httpx

from dct.config import RuntimeSettings
from dct.llm.provider import OpenAICompatibleProvider


def test_openai_provider_fallback_when_response_format_unsupported(monkeypatch):
    settings = RuntimeSettings(
        model_provider="openai_compatible",
        openai_base_url="http://localhost:11434/v1",
        openai_api_key="ollama",
        model_name="dummy-model",
        model_temperature=0.1,
    )
    provider = OpenAICompatibleProvider(settings)
    payloads = []

    def fake_post(url, headers=None, json=None):  # noqa: ANN001
        payloads.append(json)
        req = httpx.Request("POST", url)
        if len(payloads) == 1:
            return httpx.Response(
                400,
                request=req,
                json={"error": {"message": "response_format is not supported by this endpoint"}},
            )
        return httpx.Response(
            200,
            request=req,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"hypotheses":[{"rule_text":"r","expression":"x+1","rationale":"ok","confidence":0.9}]}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(provider.client, "post", fake_post)
    out = provider.generate_json("Trajectory A", "{}")
    assert out["hypotheses"][0]["expression"] == "x+1"
    assert "response_format" in payloads[0]
    assert "response_format" not in payloads[1]


def test_openai_provider_accepts_reasoning_content_when_content_empty(monkeypatch):
    settings = RuntimeSettings(
        model_provider="deepseek",
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="test-key",
        model_name="deepseek-reasoner",
        model_temperature=0.1,
    )
    provider = OpenAICompatibleProvider(settings)

    def fake_post(url, headers=None, json=None):  # noqa: ANN001
        req = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=req,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": (
                                '{"hypotheses":[{"rule_text":"r","expression":"x+2","rationale":"ok","confidence":0.8}]}'
                            ),
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(provider.client, "post", fake_post)
    out = provider.generate_json("Trajectory A", "{}")
    assert out["hypotheses"][0]["expression"] == "x+2"
