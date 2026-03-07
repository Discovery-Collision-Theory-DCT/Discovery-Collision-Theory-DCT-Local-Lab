from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


OPENAI_COMPATIBLE_PROVIDERS = {
    "openai_compatible",
    "openai",
    "azure_openai",
    "xai",
    "deepseek",
    "groq",
    "mistral",
    "together",
    "fireworks",
    "openrouter",
    "ollama",
    "lmstudio",
    "vllm",
    "llamacpp",
}

NATIVE_PROVIDERS = {"anthropic", "gemini"}
SUPPORTED_PROVIDERS = OPENAI_COMPATIBLE_PROVIDERS | NATIVE_PROVIDERS


class RuntimeSettings(BaseModel):
    model_provider: str = "openai_compatible"

    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "ollama"

    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_api_key: str = ""

    google_base_url: str = "https://generativelanguage.googleapis.com"
    google_api_key: str = ""

    model_name: str = "llama3.2:3b"
    model_temperature: float = 0.2
    model_timeout_seconds: int = 60

    model_access_mode: str = "local"
    allow_remote_inference: bool = False

    dct_output_dir: Path = Path("outputs")
    dct_sqlite_path: Path = Path("outputs/dct_memory.db")
    dct_check_model_on_start: bool = True

    def __init__(self, **data):
        if not data:
            load_dotenv()
            data = self._read_env()
        else:
            load_dotenv()
            env_data = self._read_env()
            env_data.update(data)
            data = env_data
        super().__init__(**data)

    @staticmethod
    def _read_env() -> dict:
        return {
            "model_provider": os.getenv("MODEL_PROVIDER", "openai_compatible").strip().lower(),
            "openai_base_url": os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            "openai_api_key": os.getenv("OPENAI_API_KEY", "ollama"),
            "anthropic_base_url": os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
            "google_base_url": os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com"),
            "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
            "model_name": os.getenv("MODEL_NAME", "llama3.2:3b"),
            "model_temperature": float(os.getenv("MODEL_TEMPERATURE", "0.2")),
            "model_timeout_seconds": int(os.getenv("MODEL_TIMEOUT_SECONDS", "60")),
            "model_access_mode": os.getenv("MODEL_ACCESS_MODE", "local").strip().lower(),
            "allow_remote_inference": os.getenv("ALLOW_REMOTE_INFERENCE", "false").strip().lower() == "true",
            "dct_output_dir": Path(os.getenv("DCT_OUTPUT_DIR", "outputs")),
            "dct_sqlite_path": Path(os.getenv("DCT_SQLITE_PATH", "outputs/dct_memory.db")),
            "dct_check_model_on_start": os.getenv("DCT_CHECK_MODEL_ON_START", "true").lower() == "true",
        }

    def normalized_provider(self) -> str:
        return self.model_provider.strip().lower()

    def is_supported_provider(self) -> bool:
        return self.normalized_provider() in SUPPORTED_PROVIDERS

    def effective_base_url(self) -> str:
        provider = self.normalized_provider()
        if provider in OPENAI_COMPATIBLE_PROVIDERS:
            return self.openai_base_url
        if provider == "anthropic":
            return self.anthropic_base_url
        if provider == "gemini":
            return self.google_base_url
        return self.openai_base_url

    def active_api_key(self) -> str:
        provider = self.normalized_provider()
        if provider in OPENAI_COMPATIBLE_PROVIDERS:
            return self.openai_api_key
        if provider == "anthropic":
            return self.anthropic_api_key
        if provider == "gemini":
            return self.google_api_key
        return self.openai_api_key

    def is_remote_endpoint(self) -> bool:
        parsed = urlparse(self.effective_base_url())
        host = (parsed.hostname or "").strip().lower()
        if host in {"localhost", "127.0.0.1", "::1"}:
            return False
        return bool(host)

    def validate_provider_config(self) -> Optional[str]:
        provider = self.normalized_provider()
        if not self.is_supported_provider():
            supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
            return f"MODEL_PROVIDER unsupported: {provider}. Supported: {supported}"

        if provider == "anthropic" and not self.anthropic_api_key.strip():
            return "ANTHROPIC_API_KEY is required when MODEL_PROVIDER=anthropic."

        if provider == "gemini" and not self.google_api_key.strip():
            return "GOOGLE_API_KEY is required when MODEL_PROVIDER=gemini."

        return None

    def validate_model_access_policy(self) -> Optional[str]:
        mode = self.model_access_mode.strip().lower()
        if mode not in {"local", "online"}:
            return "MODEL_ACCESS_MODE must be 'local' or 'online'."

        provider_error = self.validate_provider_config()
        if provider_error:
            return provider_error

        using_remote = self.is_remote_endpoint()
        if using_remote and mode != "online":
            return (
                "Remote endpoint detected but MODEL_ACCESS_MODE is not 'online'. "
                "Set MODEL_ACCESS_MODE=online."
            )

        if using_remote and not self.allow_remote_inference:
            return (
                "Remote endpoint detected but ALLOW_REMOTE_INFERENCE=false. "
                "Set ALLOW_REMOTE_INFERENCE=true (or pass --allow-remote-inference)."
            )
        return None


class AblationConfig(BaseModel):
    no_collision: bool = False
    no_memory_write_back: bool = False
    no_verifier: bool = False
    single_verifier_mode_only: Optional[str] = None


class ExperimentConfig(BaseModel):
    name: str = "quickstart"
    seed: int = 123
    trials: int = 1
    rounds: int = 2
    hypotheses_per_trajectory: int = 2
    baselines: list[str] = [
        "baseline_single_a",
        "baseline_single_b",
        "baseline_merged_naive",
        "full_dct",
    ]
    benchmark_families: list[str] = ["symbolic", "dynamical", "compression"]
    samples_per_task_train: int = 18
    samples_per_task_heldout: int = 10
    verifier_modes: list[str] = ["predictive", "symbolic", "simulation"]
    ablation: AblationConfig = AblationConfig()
    output_dir: Path = Path("outputs/quickstart")


def load_experiment_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ExperimentConfig.model_validate(data)
