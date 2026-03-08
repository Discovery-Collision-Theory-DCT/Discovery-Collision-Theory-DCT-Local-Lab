from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from dotenv import load_dotenv

from dct.api import create_app
from dct.config import ExperimentConfig, RuntimeSettings, load_experiment_config
from dct.llm import LLMProvider, build_provider
from dct.memory import SQLiteMemory
from dct.orchestration import DCTOrchestrator

app = typer.Typer(add_completion=False, help="Discovery Collision Theory local experiment CLI")


def _default_config_for_mode(mode: str) -> Path:
    if mode == "quickstart":
        return Path("config/quickstart.yaml")
    if mode == "full":
        return Path("config/full_experiment.yaml")
    if mode == "openworld":
        return Path("config/openworld_pathfinder.yaml")
    raise ValueError(f"Unsupported mode: {mode}")


def _provider_env_hint(settings: RuntimeSettings) -> str:
    provider = settings.normalized_provider()
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    if provider == "gemini":
        return "GOOGLE_API_KEY"
    return "OPENAI_API_KEY"


def _check_model_or_exit(settings: RuntimeSettings, provider: LLMProvider) -> None:
    ok, message = provider.check_health()
    if ok:
        typer.echo(message)
        return

    typer.echo("Model health check failed.")
    typer.echo(message)
    if settings.is_remote_endpoint():
        typer.echo("Suggested fixes for online API:")
        typer.echo("  1) Check provider and endpoint are correct")
        typer.echo(f"  2) Check {_provider_env_hint(settings)} is valid and has access to MODEL_NAME")
        typer.echo("  3) Check endpoint compatibility for selected MODEL_PROVIDER")
        typer.echo("  4) Re-run command")
    else:
        typer.echo("Suggested fixes for local endpoint (Ollama example):")
        typer.echo("  1) Start server: ollama serve")
        typer.echo(f"  2) Pull model: ollama pull {settings.model_name}")
        typer.echo("  3) Confirm endpoint: curl http://localhost:11434/v1/models")
        typer.echo("  4) Re-run command")
    raise typer.Exit(code=1)


def _build_runtime_settings(
    model_provider: Optional[str],
    model_access_mode: Optional[str],
    model_name: Optional[str],
    allow_remote_inference: bool,
    openai_base_url: Optional[str],
    openai_api_key: Optional[str],
    anthropic_base_url: Optional[str],
    anthropic_api_key: Optional[str],
    google_base_url: Optional[str],
    google_api_key: Optional[str],
) -> RuntimeSettings:
    overrides = {}
    if model_provider is not None:
        overrides["model_provider"] = model_provider
    if model_access_mode is not None:
        overrides["model_access_mode"] = model_access_mode
    if model_name is not None:
        overrides["model_name"] = model_name
    if allow_remote_inference:
        overrides["allow_remote_inference"] = True

    if openai_base_url is not None:
        overrides["openai_base_url"] = openai_base_url
    if openai_api_key is not None:
        overrides["openai_api_key"] = openai_api_key

    if anthropic_base_url is not None:
        overrides["anthropic_base_url"] = anthropic_base_url
    if anthropic_api_key is not None:
        overrides["anthropic_api_key"] = anthropic_api_key

    if google_base_url is not None:
        overrides["google_base_url"] = google_base_url
    if google_api_key is not None:
        overrides["google_api_key"] = google_api_key

    return RuntimeSettings(**overrides)


def _enforce_access_policy_or_exit(settings: RuntimeSettings) -> None:
    policy_error = settings.validate_model_access_policy()
    if policy_error is None:
        return
    typer.echo("Model access policy violation.")
    typer.echo(policy_error)
    typer.echo("Current settings:")
    typer.echo(f"  MODEL_PROVIDER={settings.model_provider}")
    typer.echo(f"  MODEL_ACCESS_MODE={settings.model_access_mode}")
    typer.echo(f"  EFFECTIVE_BASE_URL={settings.effective_base_url()}")
    typer.echo(f"  ALLOW_REMOTE_INFERENCE={settings.allow_remote_inference}")
    raise typer.Exit(code=1)


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, help="Path to YAML config"),
    mode: str = typer.Option("quickstart", help="quickstart, full, or openworld"),
    output_dir: Optional[Path] = typer.Option(None, help="Override output directory"),
    skip_model_check: bool = typer.Option(False, help="Skip startup model health check"),
    model_provider: Optional[str] = typer.Option(None, help="Provider: openai_compatible/openai/anthropic/gemini/..."),
    model_access_mode: Optional[str] = typer.Option(None, help="Model access mode: local or online"),
    model_name: Optional[str] = typer.Option(None, help="Override MODEL_NAME"),
    allow_remote_inference: bool = typer.Option(False, help="Allow remote inference when using online endpoint"),
    openai_base_url: Optional[str] = typer.Option(None, help="Override OPENAI_BASE_URL"),
    openai_api_key: Optional[str] = typer.Option(None, help="Override OPENAI_API_KEY"),
    anthropic_base_url: Optional[str] = typer.Option(None, help="Override ANTHROPIC_BASE_URL"),
    anthropic_api_key: Optional[str] = typer.Option(None, help="Override ANTHROPIC_API_KEY"),
    google_base_url: Optional[str] = typer.Option(None, help="Override GOOGLE_BASE_URL"),
    google_api_key: Optional[str] = typer.Option(None, help="Override GOOGLE_API_KEY"),
) -> None:
    """Run benchmark experiment and write JSON/CSV/plots to output dir."""
    load_dotenv()
    settings = _build_runtime_settings(
        model_provider=model_provider,
        model_access_mode=model_access_mode,
        model_name=model_name,
        allow_remote_inference=allow_remote_inference,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        anthropic_base_url=anthropic_base_url,
        anthropic_api_key=anthropic_api_key,
        google_base_url=google_base_url,
        google_api_key=google_api_key,
    )
    _enforce_access_policy_or_exit(settings)

    config_path = config or _default_config_for_mode(mode)
    if not config_path.exists():
        typer.echo(f"Config file not found: {config_path}")
        raise typer.Exit(code=1)

    exp_config: ExperimentConfig = load_experiment_config(config_path)
    if output_dir is not None:
        exp_config.output_dir = output_dir

    provider = build_provider(settings)
    if settings.dct_check_model_on_start and not skip_model_check:
        _check_model_or_exit(settings, provider)

    memory = SQLiteMemory(settings.dct_sqlite_path)
    orchestrator = DCTOrchestrator(settings=settings, provider=provider, memory=memory)

    try:
        summary, run_output_dir = orchestrator.run(exp_config)
    except Exception as exc:  # noqa: BLE001
        typer.echo("Experiment run failed.")
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    finally:
        memory.close()

    typer.echo(f"Run complete: {summary.run_name}")
    typer.echo(f"Output directory: {run_output_dir}")
    typer.echo("Uplift (full_dct over baselines):")
    typer.echo(json.dumps(summary.uplift, indent=2))


@app.command("quickstart")
def quickstart() -> None:
    """Run the lightweight quickstart experiment."""
    run(
        config=Path("config/quickstart.yaml"),
        mode="quickstart",
        output_dir=None,
        skip_model_check=False,
        model_provider=None,
        model_access_mode=None,
        model_name=None,
        allow_remote_inference=False,
        openai_base_url=None,
        openai_api_key=None,
        anthropic_base_url=None,
        anthropic_api_key=None,
        google_base_url=None,
        google_api_key=None,
    )


@app.command("openworld")
def openworld() -> None:
    """Run the open-world pathfinder experiment pack."""
    run(
        config=Path("config/openworld_pathfinder.yaml"),
        mode="openworld",
        output_dir=None,
        skip_model_check=False,
        model_provider=None,
        model_access_mode=None,
        model_name=None,
        allow_remote_inference=False,
        openai_base_url=None,
        openai_api_key=None,
        anthropic_base_url=None,
        anthropic_api_key=None,
        google_base_url=None,
        google_api_key=None,
    )


@app.command("check-model")
def check_model(
    model_provider: Optional[str] = typer.Option(None, help="Provider: openai_compatible/openai/anthropic/gemini/..."),
    model_access_mode: Optional[str] = typer.Option(None, help="Model access mode: local or online"),
    model_name: Optional[str] = typer.Option(None, help="Override MODEL_NAME"),
    allow_remote_inference: bool = typer.Option(False, help="Allow remote inference when using online endpoint"),
    openai_base_url: Optional[str] = typer.Option(None, help="Override OPENAI_BASE_URL"),
    openai_api_key: Optional[str] = typer.Option(None, help="Override OPENAI_API_KEY"),
    anthropic_base_url: Optional[str] = typer.Option(None, help="Override ANTHROPIC_BASE_URL"),
    anthropic_api_key: Optional[str] = typer.Option(None, help="Override ANTHROPIC_API_KEY"),
    google_base_url: Optional[str] = typer.Option(None, help="Override GOOGLE_BASE_URL"),
    google_api_key: Optional[str] = typer.Option(None, help="Override GOOGLE_API_KEY"),
) -> None:
    """Check whether the configured model endpoint is reachable."""
    load_dotenv()
    settings = _build_runtime_settings(
        model_provider=model_provider,
        model_access_mode=model_access_mode,
        model_name=model_name,
        allow_remote_inference=allow_remote_inference,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        anthropic_base_url=anthropic_base_url,
        anthropic_api_key=anthropic_api_key,
        google_base_url=google_base_url,
        google_api_key=google_api_key,
    )
    _enforce_access_policy_or_exit(settings)
    provider = build_provider(settings)
    ok, message = provider.check_health()
    typer.echo(message)
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def serve(
    output_root: Path = typer.Option(Path("outputs"), help="Root directory that contains run outputs"),
    host: str = typer.Option("127.0.0.1", help="Host"),
    port: int = typer.Option(8000, help="Port"),
) -> None:
    """Serve optional local API for browsing run summaries."""
    api = create_app(output_root)
    uvicorn.run(api, host=host, port=port)


if __name__ == "__main__":
    app()
