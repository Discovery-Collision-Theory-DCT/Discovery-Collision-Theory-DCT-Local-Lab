from dct.config import RuntimeSettings


def test_local_mode_with_localhost_is_valid():
    settings = RuntimeSettings(
        model_access_mode="local",
        openai_base_url="http://localhost:11434/v1",
        allow_remote_inference=False,
    )
    assert settings.validate_model_access_policy() is None


def test_remote_endpoint_requires_online_mode():
    settings = RuntimeSettings(
        model_access_mode="local",
        openai_base_url="https://api.openai.com/v1",
        allow_remote_inference=True,
    )
    msg = settings.validate_model_access_policy()
    assert msg is not None
    assert "MODEL_ACCESS_MODE=online" in msg


def test_remote_endpoint_requires_allow_remote_inference():
    settings = RuntimeSettings(
        model_access_mode="online",
        openai_base_url="https://api.openai.com/v1",
        allow_remote_inference=False,
    )
    msg = settings.validate_model_access_policy()
    assert msg is not None
    assert "ALLOW_REMOTE_INFERENCE=true" in msg


def test_online_remote_with_permission_is_valid():
    settings = RuntimeSettings(
        model_access_mode="online",
        openai_base_url="https://api.openai.com/v1",
        allow_remote_inference=True,
    )
    assert settings.validate_model_access_policy() is None
