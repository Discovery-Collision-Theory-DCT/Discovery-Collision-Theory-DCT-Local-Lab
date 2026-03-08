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

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: ANN001
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

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: ANN001
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


def test_openai_provider_accepts_choice_text_fallback(monkeypatch):
    settings = RuntimeSettings(
        model_provider="openai_compatible",
        openai_base_url="http://localhost:11434/v1",
        openai_api_key="ollama",
        model_name="dummy-model",
        model_temperature=0.1,
    )
    provider = OpenAICompatibleProvider(settings)

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: ANN001
        req = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=req,
            json={
                "choices": [
                    {
                        "text": '{"hypotheses":[{"rule_text":"r","expression":"x+3","rationale":"ok","confidence":0.7}]}'
                    }
                ]
            },
        )

    monkeypatch.setattr(provider.client, "post", fake_post)
    out = provider.generate_json("Trajectory A", "{}")
    assert out["hypotheses"][0]["expression"] == "x+3"


def test_openai_provider_reasoner_uses_larger_max_tokens(monkeypatch):
    settings = RuntimeSettings(
        model_provider="deepseek",
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="test-key",
        model_name="deepseek-reasoner",
        model_temperature=0.1,
    )
    provider = OpenAICompatibleProvider(settings)
    seen_max_tokens = []

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: ANN001
        seen_max_tokens.append(json.get("max_tokens"))
        req = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=req,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"hypotheses":[{"rule_text":"r","expression":"x+4","rationale":"ok","confidence":0.6}]}'
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(provider.client, "post", fake_post)
    out = provider.generate_json("Trajectory A", "{}")
    assert out["hypotheses"][0]["expression"] == "x+4"
    assert seen_max_tokens
    assert seen_max_tokens[0] >= 1600


def test_openai_provider_repair_can_fallback_to_chat_model(monkeypatch):
    settings = RuntimeSettings(
        model_provider="deepseek",
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="test-key",
        model_name="deepseek-reasoner",
        model_temperature=0.1,
    )
    provider = OpenAICompatibleProvider(settings)
    seen_models = []

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: ANN001
        model = json.get("model")
        seen_models.append(model)
        req = httpx.Request("POST", url)
        messages = json.get("messages", [])
        is_repair = bool(messages and isinstance(messages[0], dict) and "repair malformed json" in str(messages[0].get("content", "")).lower())

        if not is_repair:
            return httpx.Response(
                200,
                request=req,
                json={"choices": [{"message": {"content": "not valid json"}}]},
            )
        if model == "deepseek-reasoner":
            return httpx.Response(
                200,
                request=req,
                json={"choices": [{"message": {"content": "still not valid"}}]},
            )
        return httpx.Response(
            200,
            request=req,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"hypotheses":[{"rule_text":"r","expression":"x+5","rationale":"ok","confidence":0.9}]}'
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(provider.client, "post", fake_post)
    out = provider.generate_json("Trajectory A", "{}")
    assert out["hypotheses"][0]["expression"] == "x+5"
    assert "deepseek-chat" in seen_models


def test_openai_provider_reasoner_uses_extended_timeout(monkeypatch):
    settings = RuntimeSettings(
        model_provider="deepseek",
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="test-key",
        model_name="deepseek-reasoner",
        model_timeout_seconds=60,
    )
    provider = OpenAICompatibleProvider(settings)
    seen_timeouts = []

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: ANN001
        seen_timeouts.append(timeout)
        req = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=req,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"hypotheses":[{"rule_text":"r","expression":"x+6","rationale":"ok","confidence":0.9}]}'
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(provider.client, "post", fake_post)
    out = provider.generate_json("Trajectory A", "{}")
    assert out["hypotheses"][0]["expression"] == "x+6"
    assert seen_timeouts
    assert seen_timeouts[0] >= 180
