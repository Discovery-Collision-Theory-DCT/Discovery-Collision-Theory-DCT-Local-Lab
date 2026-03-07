from dct.config import RuntimeSettings
from dct.llm.provider import AnthropicProvider, GeminiProvider, OpenAICompatibleProvider, build_provider


def test_build_provider_openai_compatible_aliases():
    settings = RuntimeSettings(model_provider="deepseek")
    provider = build_provider(settings)
    assert isinstance(provider, OpenAICompatibleProvider)


def test_build_provider_anthropic():
    settings = RuntimeSettings(model_provider="anthropic", anthropic_api_key="test-key")
    provider = build_provider(settings)
    assert isinstance(provider, AnthropicProvider)


def test_build_provider_gemini():
    settings = RuntimeSettings(model_provider="gemini", google_api_key="test-key")
    provider = build_provider(settings)
    assert isinstance(provider, GeminiProvider)


def test_provider_config_requires_anthropic_key():
    settings = RuntimeSettings(model_provider="anthropic", anthropic_api_key="")
    msg = settings.validate_model_access_policy()
    assert msg is not None
    assert "ANTHROPIC_API_KEY" in msg


def test_provider_config_requires_google_key():
    settings = RuntimeSettings(model_provider="gemini", google_api_key="")
    msg = settings.validate_model_access_policy()
    assert msg is not None
    assert "GOOGLE_API_KEY" in msg
