from __future__ import annotations

import json
import math
import re
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dct.config import OPENAI_COMPATIBLE_PROVIDERS, RuntimeSettings, load_experiment_config
from dct.llm import build_provider
from dct.memory import SQLiteMemory
from dct.orchestration import DCTOrchestrator
from dct.utils import clamp01, jaccard_similarity, token_set


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProviderRuntimeRequest(BaseModel):
    model_provider: str | None = None
    model_access_mode: str | None = None
    model_name: str | None = None
    model_temperature: float | None = None
    use_reasoner: bool = False
    reasoner_model_name: str | None = None
    allow_remote_inference: bool = False

    openai_base_url: str | None = None
    openai_api_key: str | None = None
    anthropic_base_url: str | None = None
    anthropic_api_key: str | None = None
    google_base_url: str | None = None
    google_api_key: str | None = None


class RunRequest(ProviderRuntimeRequest):
    config_path: str | None = None
    mode: str = "quickstart"
    output_dir: str | None = None
    skip_model_check: bool = False


class ProviderModelsRequest(ProviderRuntimeRequest):
    pass


class RunExplainRequest(ProviderRuntimeRequest):
    focus: str | None = None


class DiscoveryVector(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class DiscoveryInput(BaseModel):
    discovery_id: str | None = None
    title: str = ""
    expression: str
    rationale: str = ""
    confidence: float = 0.5
    direction: DiscoveryVector = Field(default_factory=DiscoveryVector)


class DiscoveryCollisionRequest(ProviderRuntimeRequest):
    discoveries: list[DiscoveryInput]
    known_theories: list[str] = Field(default_factory=list)
    memory_expressions: list[str] = Field(default_factory=list)
    max_collisions: int = 4


class RunJob(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    request: dict
    error: str | None = None
    run_name: str | None = None
    run_output_dir: str | None = None
    logs: list[dict] = Field(default_factory=list)
    last_event: dict | None = None


class JobCancelledError(RuntimeError):
    pass


def _vector_tuple(direction: DiscoveryVector) -> tuple[float, float, float]:
    return float(direction.x), float(direction.y), float(direction.z)


def _vector_norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _directional_complementarity(a: DiscoveryVector, b: DiscoveryVector) -> float:
    va = _vector_tuple(a)
    vb = _vector_tuple(b)
    na = _vector_norm(va)
    nb = _vector_norm(vb)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.5
    cosine = (va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]) / (na * nb)
    cosine = max(-1.0, min(1.0, cosine))
    return clamp01((1.0 - cosine) / 2.0)


def _expression_novelty(expression: str, reference_expressions: list[str]) -> float:
    if not reference_expressions:
        return 1.0
    expr_tokens = token_set(expression)
    if not expr_tokens:
        return 0.0
    sims = [jaccard_similarity(expr_tokens, token_set(ref)) for ref in reference_expressions if ref.strip()]
    if not sims:
        return 1.0
    return clamp01(1.0 - max(sims))


def _is_online_runtime(settings: RuntimeSettings) -> bool:
    return (
        settings.model_access_mode.strip().lower() == "online"
        and settings.allow_remote_inference
        and settings.is_remote_endpoint()
    )


def _sanitize_expression_list(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        expr = str(value or "").strip()
        if not expr:
            continue
        if expr in seen:
            continue
        seen.add(expr)
        out.append(expr)
    return out


def _coerce_confidence(value: Any, default: float = 0.5) -> float:
    try:
        return clamp01(float(value))
    except (TypeError, ValueError):
        return default


def _condense_method_summaries(method_summaries: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in method_summaries:
        method = str(row.get("method", "")).strip()
        if not method:
            continue
        grouped[method].append(row)

    metrics = [
        "validity_rate",
        "heldout_predictive_accuracy",
        "ood_predictive_accuracy",
        "stress_predictive_accuracy",
        "transfer_generalization_score",
        "open_world_readiness_score",
        "rule_recovery_exact_match_rate",
        "cumulative_improvement",
    ]

    condensed: list[dict] = []
    for method, rows in sorted(grouped.items(), key=lambda x: x[0]):
        item: dict[str, Any] = {"method": method, "trial_count": len(rows)}
        for metric in metrics:
            vals: list[float] = []
            for row in rows:
                try:
                    vals.append(float(row.get(metric, 0.0)))
                except (TypeError, ValueError):
                    continue
            item[metric] = float(sum(vals) / len(vals)) if vals else 0.0
        condensed.append(item)
    return condensed


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _normalize_model_id(provider: str, raw_model_id: str) -> str:
    model_id = str(raw_model_id or "").strip()
    if not model_id:
        return ""
    if provider == "gemini" and model_id.startswith("models/"):
        return model_id.split("/", 1)[1]
    return model_id


def _redact_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except Exception:  # noqa: BLE001
        return url
    query_pairs = parse_qsl(parts.query, keep_blank_values=True)
    redacted = []
    for key, value in query_pairs:
        if key.lower() in {"key", "api_key", "token"} and value:
            redacted.append((key, "***"))
        else:
            redacted.append((key, value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(redacted), parts.fragment))


def _looks_like_reasoner_model(model_name: str) -> bool:
    low = str(model_name or "").strip().lower()
    if not low:
        return False
    if any(marker in low for marker in ["reasoner", "reasoning", "thinking", "deepseek-r1", "r1-"]):
        return True
    if re.search(r"(^|[-_/])r1($|[-_/])", low):
        return True
    if re.search(r"(^|[-_/])o[13]($|[-_/])", low):
        return True
    if re.search(r"claude-.*thinking", low):
        return True
    return False


def _split_reasoner_models(models: list[str]) -> tuple[list[str], list[str]]:
    deduped = _dedupe_preserve_order(models)
    reasoners = [m for m in deduped if _looks_like_reasoner_model(m)]
    if not reasoners:
        reasoners = deduped[: min(12, len(deduped))]
    return deduped, reasoners


def _extract_models_from_openai_payload(payload: dict[str, Any], provider: str) -> list[str]:
    rows = payload.get("data")
    if not isinstance(rows, list):
        rows = payload.get("models")
    if not isinstance(rows, list):
        rows = []

    out: list[str] = []
    for row in rows:
        if isinstance(row, str):
            model_id = _normalize_model_id(provider, row)
            if model_id:
                out.append(model_id)
            continue
        if not isinstance(row, dict):
            continue
        model_id = row.get("id") or row.get("name") or row.get("model")
        model_id = _normalize_model_id(provider, str(model_id or ""))
        if model_id:
            out.append(model_id)
    return _dedupe_preserve_order(out)


def _extract_models_from_anthropic_payload(payload: dict[str, Any], provider: str) -> list[str]:
    rows = payload.get("data")
    if not isinstance(rows, list):
        rows = payload.get("models")
    if not isinstance(rows, list):
        rows = []

    out: list[str] = []
    for row in rows:
        if isinstance(row, str):
            model_id = _normalize_model_id(provider, row)
            if model_id:
                out.append(model_id)
            continue
        if not isinstance(row, dict):
            continue
        model_id = row.get("id") or row.get("name")
        model_id = _normalize_model_id(provider, str(model_id or ""))
        if model_id:
            out.append(model_id)
    return _dedupe_preserve_order(out)


def _extract_models_from_gemini_payload(payload: dict[str, Any], provider: str) -> list[str]:
    rows = payload.get("models")
    if not isinstance(rows, list):
        rows = payload.get("data")
    if not isinstance(rows, list):
        rows = []

    out: list[str] = []
    for row in rows:
        if isinstance(row, str):
            model_id = _normalize_model_id(provider, row)
            if model_id:
                out.append(model_id)
            continue
        if not isinstance(row, dict):
            continue
        model_id = row.get("name") or row.get("id")
        model_id = _normalize_model_id(provider, str(model_id or ""))
        if model_id:
            out.append(model_id)
    return _dedupe_preserve_order(out)


def _fetch_available_models(settings: RuntimeSettings) -> tuple[list[str], str]:
    provider = settings.normalized_provider()
    timeout = max(8.0, min(60.0, float(settings.model_timeout_seconds)))

    with httpx.Client(timeout=timeout) as client:
        if provider in OPENAI_COMPATIBLE_PROVIDERS:
            url = f"{settings.openai_base_url.rstrip('/')}/models"
            response = client.get(
                url,
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            payload = response.json()
            models = _extract_models_from_openai_payload(payload, provider)
            return models, _redact_url(url)

        if provider == "anthropic":
            if not settings.anthropic_api_key.strip():
                raise ValueError("ANTHROPIC_API_KEY is required for anthropic model listing.")
            url = f"{settings.anthropic_base_url.rstrip('/')}/v1/models"
            response = client.get(
                url,
                headers={
                    "x-api-key": settings.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            payload = response.json()
            models = _extract_models_from_anthropic_payload(payload, provider)
            return models, _redact_url(url)

        if provider == "gemini":
            if not settings.google_api_key.strip():
                raise ValueError("GOOGLE_API_KEY is required for gemini model listing.")
            url = f"{settings.google_base_url.rstrip('/')}/v1beta/models?key={settings.google_api_key}"
            response = client.get(url)
            response.raise_for_status()
            payload = response.json()
            models = _extract_models_from_gemini_payload(payload, provider)
            return models, _redact_url(url)

    raise ValueError(f"Unsupported MODEL_PROVIDER for model listing: {provider}")


def create_app(output_root: Path) -> FastAPI:
    output_root = output_root.resolve()
    ui_root = Path(__file__).resolve().parents[1] / "ui"
    repo_root = Path(__file__).resolve().parents[2]
    jobs_state_path = output_root / ".ui_jobs_state.json"

    app = FastAPI(title="DCT Local Dashboard API", version="0.3.0")
    app.mount("/outputs", StaticFiles(directory=str(output_root), check_dir=False), name="outputs")
    if ui_root.exists():
        app.mount("/ui-assets", StaticFiles(directory=str(ui_root), check_dir=True), name="ui-assets")

    jobs: dict[str, RunJob] = {}
    job_cancel_flags: dict[str, threading.Event] = {}
    jobs_lock = threading.Lock()
    model_output_max_chars = 4000

    def _persist_jobs_locked() -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": utc_now(),
            "jobs": [job.model_dump(mode="json") for job in jobs.values()],
        }
        tmp_path = jobs_state_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(jobs_state_path)

    def _load_jobs_from_disk() -> None:
        if not jobs_state_path.exists():
            return
        try:
            data = json.loads(jobs_state_path.read_text(encoding="utf-8"))
            rows = data.get("jobs", [])
            if not isinstance(rows, list):
                return
            for item in rows:
                try:
                    job = RunJob.model_validate(item)
                except Exception:  # noqa: BLE001
                    continue
                jobs[job.job_id] = job
        except Exception:  # noqa: BLE001
            return

    _load_jobs_from_disk()

    def _default_config_for_mode(mode: str) -> Path:
        if mode == "quickstart":
            return repo_root / "config" / "quickstart.yaml"
        if mode == "full":
            return repo_root / "config" / "full_experiment.yaml"
        if mode == "openworld":
            return repo_root / "config" / "openworld_pathfinder.yaml"
        raise ValueError(f"Unsupported mode: {mode}")

    def _build_runtime_settings(req: ProviderRuntimeRequest) -> RuntimeSettings:
        overrides: dict[str, Any] = {}
        if req.model_provider is not None:
            overrides["model_provider"] = req.model_provider
        if req.model_access_mode is not None:
            overrides["model_access_mode"] = req.model_access_mode
        if req.model_name is not None:
            overrides["model_name"] = req.model_name
        if req.model_temperature is not None:
            overrides["model_temperature"] = req.model_temperature
        if req.use_reasoner:
            overrides["model_name"] = req.reasoner_model_name or "deepseek-reasoner"
        if req.allow_remote_inference:
            overrides["allow_remote_inference"] = True

        if req.openai_base_url is not None:
            overrides["openai_base_url"] = req.openai_base_url
        if req.openai_api_key is not None:
            overrides["openai_api_key"] = req.openai_api_key
        if req.anthropic_base_url is not None:
            overrides["anthropic_base_url"] = req.anthropic_base_url
        if req.anthropic_api_key is not None:
            overrides["anthropic_api_key"] = req.anthropic_api_key
        if req.google_base_url is not None:
            overrides["google_base_url"] = req.google_base_url
        if req.google_api_key is not None:
            overrides["google_api_key"] = req.google_api_key

        return RuntimeSettings(**overrides)

    def _summary_files() -> list[Path]:
        if not output_root.exists():
            return []
        return sorted(output_root.glob("**/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    def _run_entry(summary_file: Path) -> dict:
        return {
            "run_name": summary_file.parent.name,
            "summary_path": str(summary_file),
            "updated_at": datetime.fromtimestamp(summary_file.stat().st_mtime, tz=timezone.utc).isoformat(),
            "run_dir": str(summary_file.parent),
        }

    def _find_summary(run_name: str) -> Path:
        candidates = list(output_root.glob(f"**/{run_name}/summary.json"))
        if not candidates:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_name}")
        return candidates[0]

    def _update_job(job_id: str, **kwargs) -> None:
        with jobs_lock:
            if job_id not in jobs:
                return
            job = jobs[job_id]
            payload = job.model_dump()
            payload.update(kwargs)
            payload["updated_at"] = utc_now()
            jobs[job_id] = RunJob.model_validate(payload)
            _persist_jobs_locked()

    def _is_cancel_requested(job_id: str) -> bool:
        with jobs_lock:
            flag = job_cancel_flags.get(job_id)
            return bool(flag and flag.is_set())

    def _append_model_output_log(job_id: str, event: dict) -> None:
        raw_text = str(event.get("text", ""))
        text = raw_text[:model_output_max_chars]
        if len(raw_text) > model_output_max_chars:
            text += "\n...[truncated]"

        event_payload = dict(event)
        event_payload["text"] = text

        provider = event_payload.get("provider", "unknown")
        model = event_payload.get("model", "unknown")
        phase = event_payload.get("phase", "primary")
        agent = event_payload.get("agent_hint", "agent")
        message = f"{provider}/{model} [{phase}] {agent}"
        _append_job_log(job_id, message=message, level="model", event=event_payload)

    def _append_job_log(
        job_id: str,
        message: str,
        level: str = "info",
        event: dict | None = None,
    ) -> None:
        with jobs_lock:
            if job_id not in jobs:
                return
            job = jobs[job_id]
            payload = job.model_dump()
            logs = list(payload.get("logs", []))
            log_item = {
                "time": utc_now(),
                "level": level,
                "message": message,
            }
            if event is not None:
                log_item["event"] = event
                payload["last_event"] = event
            logs.append(log_item)
            payload["logs"] = logs[-500:]
            payload["updated_at"] = utc_now()
            jobs[job_id] = RunJob.model_validate(payload)
            _persist_jobs_locked()

    def _event_to_log_message(event: dict) -> str:
        event_type = event.get("type", "event")
        if event_type == "run_started":
            return (
                f"Run started: name={event.get('run_name')} baselines={len(event.get('baselines', []))} "
                f"trials={event.get('trials')} rounds={event.get('rounds')}"
            )
        if event_type == "trial_started":
            return f"Trial started: method={event.get('method')} trial={event.get('trial_index')}"
        if event_type == "task_started":
            return (
                f"Task started: method={event.get('method')} trial={event.get('trial_index')} "
                f"round={event.get('round_index')} family={event.get('family')}"
            )
        if event_type == "task_completed":
            validity = event.get("validity_rate")
            top_acc = event.get("top_heldout_accuracy")
            top_ood = event.get("top_ood_accuracy")
            top_stress = event.get("top_stress_accuracy")
            try:
                validity_str = f"{float(validity):.3f}"
            except (TypeError, ValueError):
                validity_str = "n/a"
            try:
                top_acc_str = f"{float(top_acc):.3f}"
            except (TypeError, ValueError):
                top_acc_str = "n/a"
            try:
                top_ood_str = f"{float(top_ood):.3f}"
            except (TypeError, ValueError):
                top_ood_str = "n/a"
            try:
                top_stress_str = f"{float(top_stress):.3f}"
            except (TypeError, ValueError):
                top_stress_str = "n/a"
            return (
                f"Task completed: method={event.get('method')} trial={event.get('trial_index')} "
                f"round={event.get('round_index')} family={event.get('family')} "
                f"candidates={event.get('candidate_count')} accepted={event.get('accepted_count')} "
                f"validity={validity_str} top_acc={top_acc_str} "
                f"ood={top_ood_str} stress={top_stress_str}"
            )
        if event_type == "trial_completed":
            return (
                f"Trial completed: method={event.get('method')} trial={event.get('trial_index')} "
                f"candidates={event.get('candidate_count')} accepted={event.get('accepted_count')}"
            )
        if event_type == "run_completed":
            return f"Run completed: name={event.get('run_name')} output={event.get('run_output_dir')}"
        return f"Event: {event_type}"

    def _execute_run_job(job_id: str, req: RunRequest) -> None:
        def _ensure_not_cancelled() -> None:
            if _is_cancel_requested(job_id):
                raise JobCancelledError("Job cancelled by user")

        if _is_cancel_requested(job_id):
            _update_job(job_id, status="cancelled", error=None)
            _append_job_log(job_id, "Job cancelled before start", level="warning")
            return

        _update_job(job_id, status="running")
        _append_job_log(job_id, "Job running")

        memory = None
        try:
            _ensure_not_cancelled()
            settings = _build_runtime_settings(req)
            policy_error = settings.validate_model_access_policy()
            if policy_error:
                raise RuntimeError(policy_error)
            _append_job_log(
                job_id,
                (
                    "Settings loaded: "
                    f"provider={settings.model_provider} model={settings.model_name} "
                    f"temperature={settings.model_temperature} access_mode={settings.model_access_mode}"
                ),
            )
            if req.use_reasoner:
                _append_job_log(job_id, f"Reasoner enabled: model={settings.model_name}")

            _ensure_not_cancelled()
            provider = build_provider(settings)
            if hasattr(provider, "set_debug_callback"):
                provider.set_debug_callback(lambda event: _append_model_output_log(job_id, event))
            if settings.dct_check_model_on_start and not req.skip_model_check:
                _ensure_not_cancelled()
                ok, message = provider.check_health()
                if not ok:
                    raise RuntimeError(f"Model check failed: {message}")
                _append_job_log(job_id, f"Model check passed: {message}")
            else:
                _append_job_log(job_id, "Model check skipped")

            _ensure_not_cancelled()
            config_path = Path(req.config_path).resolve() if req.config_path else _default_config_for_mode(req.mode)
            if not config_path.exists():
                raise RuntimeError(f"Config file not found: {config_path}")
            _append_job_log(job_id, f"Using config: {config_path}")

            _ensure_not_cancelled()
            exp_config = load_experiment_config(config_path)
            if req.output_dir:
                exp_config.output_dir = Path(req.output_dir)
            _append_job_log(job_id, f"Output dir: {exp_config.output_dir}")

            _ensure_not_cancelled()
            memory = SQLiteMemory(settings.dct_sqlite_path)
            orchestrator = DCTOrchestrator(settings=settings, provider=provider, memory=memory)

            def _progress_callback(event: dict[str, Any]) -> None:
                _ensure_not_cancelled()
                _append_job_log(
                    job_id,
                    message=_event_to_log_message(event),
                    level="info",
                    event=event,
                )

            summary, run_output_dir = orchestrator.run(
                exp_config,
                progress_callback=_progress_callback,
            )
            _ensure_not_cancelled()

            _update_job(
                job_id,
                status="completed",
                run_name=summary.run_name,
                run_output_dir=str(run_output_dir),
                error=None,
            )
            _append_job_log(job_id, f"Job completed: run_name={summary.run_name}")
        except JobCancelledError:
            _update_job(job_id, status="cancelled", error=None)
            _append_job_log(job_id, "Job cancelled by user", level="warning")
        except Exception as exc:  # noqa: BLE001
            _update_job(job_id, status="failed", error=str(exc))
            _append_job_log(job_id, f"Job failed: {exc}", level="error")
        finally:
            if memory is not None:
                memory.close()
            with jobs_lock:
                job_cancel_flags.pop(job_id, None)

    @app.get("/")
    def ui_index():
        index_path = ui_root / "index.html"
        if not index_path.exists():
            return {
                "message": "UI assets not found",
                "hint": "Expected dct/ui/index.html",
            }
        return FileResponse(index_path)

    @app.get("/health")
    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "time": utc_now()}

    @app.get("/runs")
    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        return [_run_entry(sf) for sf in _summary_files()]

    @app.get("/runs/{run_name}")
    @app.get("/api/runs/{run_name}")
    def get_run(run_name: str) -> dict:
        summary_path = _find_summary(run_name)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    @app.get("/latest")
    @app.get("/api/latest")
    def latest() -> dict:
        summaries = _summary_files()
        if not summaries:
            raise HTTPException(status_code=404, detail="No experiment summaries found")
        return json.loads(summaries[0].read_text(encoding="utf-8"))

    @app.get("/api/runs/{run_name}/artifacts")
    def run_artifacts(run_name: str) -> dict:
        summary_path = _find_summary(run_name)
        run_dir = summary_path.parent

        artifacts = []
        for file_path in sorted(run_dir.rglob("*")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(output_root).as_posix()
            artifacts.append(
                {
                    "name": file_path.name,
                    "path": str(file_path),
                    "relative_path": rel,
                    "url": f"/outputs/{rel}",
                    "size_bytes": file_path.stat().st_size,
                }
            )

        return {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "artifacts": artifacts,
        }

    @app.post("/api/runs/{run_name}/explain")
    def explain_run(run_name: str, req: RunExplainRequest) -> dict:
        summary_path = _find_summary(run_name)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        settings = _build_runtime_settings(req)
        policy_error = settings.validate_model_access_policy()
        if policy_error:
            raise HTTPException(status_code=400, detail=policy_error)
        if not _is_online_runtime(settings):
            raise HTTPException(
                status_code=400,
                detail=(
                    "LLM explanation is only available for online remote providers. "
                    "Set model_access_mode=online, allow_remote_inference=true, and a non-local base URL."
                ),
            )

        method_summaries = summary.get("method_summaries", [])
        condensed = _condense_method_summaries(method_summaries if isinstance(method_summaries, list) else [])

        payload = {
            "run_name": summary.get("run_name", run_name),
            "focus": (req.focus or "overall_dct_performance").strip() or "overall_dct_performance",
            "condensed_method_summaries": condensed,
            "uplift": summary.get("uplift", {}),
            "config": summary.get("config", {}),
        }
        system_prompt = (
            "You are an experiment analyst. "
            "Explain benchmark outcomes with concrete evidence from the provided metrics. "
            "Return JSON with keys: executive_summary (string), key_findings (array of strings), "
            "risks (array of strings), recommended_next_experiments (array of strings), confidence (0..1). "
            "Do not return markdown."
        )

        provider = build_provider(settings)
        try:
            explanation = provider.generate_json(system_prompt, json.dumps(payload, ensure_ascii=True), max_tokens=1200)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Failed to generate LLM explanation: {exc}") from exc

        return {
            "run_name": run_name,
            "provider": settings.normalized_provider(),
            "model": settings.model_name,
            "focus": payload["focus"],
            "explanation": explanation,
        }

    @app.post("/api/discovery/collide")
    def discovery_collide(req: DiscoveryCollisionRequest) -> dict:
        if len(req.discoveries) < 2:
            raise HTTPException(status_code=400, detail="At least 2 discoveries are required for collision.")

        settings = _build_runtime_settings(req)
        policy_error = settings.validate_model_access_policy()
        if policy_error:
            raise HTTPException(status_code=400, detail=policy_error)

        max_collisions = max(1, min(12, int(req.max_collisions)))
        discoveries = []
        for idx, item in enumerate(req.discoveries):
            expr = str(item.expression or "").strip()
            if not expr:
                continue
            discovery_id = (item.discovery_id or f"d{idx + 1}").strip()[:80] or f"d{idx + 1}"
            discoveries.append(
                {
                    "discovery_id": discovery_id,
                    "title": (item.title or "").strip()[:120],
                    "expression": expr,
                    "rationale": (item.rationale or "").strip()[:1000],
                    "confidence": _coerce_confidence(item.confidence),
                    "direction": item.direction.model_dump(),
                    "direction_model": item.direction,
                }
            )
        if len(discoveries) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 non-empty discovery expressions.")

        pair_scores: list[dict[str, Any]] = []
        for i in range(len(discoveries)):
            for j in range(i + 1, len(discoveries)):
                left = discoveries[i]
                right = discoveries[j]
                directional = _directional_complementarity(
                    left["direction_model"],
                    right["direction_model"],
                )
                expr_diversity = clamp01(
                    1.0 - jaccard_similarity(token_set(left["expression"]), token_set(right["expression"]))
                )
                confidence_mix = clamp01((left["confidence"] + right["confidence"]) / 2.0)
                pair_strength = clamp01(
                    0.5 * directional + 0.35 * expr_diversity + 0.15 * confidence_mix
                )
                pair_scores.append(
                    {
                        "left_id": left["discovery_id"],
                        "right_id": right["discovery_id"],
                        "directional_complementarity": directional,
                        "expression_diversity": expr_diversity,
                        "confidence_mix": confidence_mix,
                        "collision_strength": pair_strength,
                    }
                )

        pair_scores = sorted(pair_scores, key=lambda p: p["collision_strength"], reverse=True)
        top_pairs = pair_scores[:max_collisions]

        reference_expressions = _sanitize_expression_list(
            [d["expression"] for d in discoveries] + req.known_theories + req.memory_expressions
        )
        llm_payload = {
            "discoveries": [
                {
                    "discovery_id": d["discovery_id"],
                    "title": d["title"],
                    "expression": d["expression"],
                    "rationale": d["rationale"],
                    "confidence": d["confidence"],
                    "direction": d["direction"],
                }
                for d in discoveries
            ],
            "top_pairs": top_pairs,
            "known_theories": _sanitize_expression_list(req.known_theories),
            "memory_expressions": _sanitize_expression_list(req.memory_expressions),
            "task": (
                "Synthesize collision hypotheses from multiple discoveries with vector directions and "
                "estimate whether each candidate is a genuinely new theory."
            ),
        }
        system_prompt = (
            "You are a scientific hypothesis synthesizer. "
            "Given discovery vectors and top collision pairs, generate high-quality collision hypotheses. "
            "Return JSON with key collision_hypotheses (array). "
            "Each item must contain: title, rule_text, expression, rationale, confidence (0..1), "
            "source_pair (array of two discovery_ids), is_new_theory (boolean), novelty_reason (string). "
            "Do not use markdown."
        )

        provider = build_provider(settings)
        try:
            llm_result = provider.generate_json(system_prompt, json.dumps(llm_payload, ensure_ascii=True), max_tokens=1400)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Collision synthesis failed: {exc}") from exc

        raw_hypotheses = llm_result.get("collision_hypotheses", [])
        if not isinstance(raw_hypotheses, list):
            raw_hypotheses = []

        synthesized: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_hypotheses[:max_collisions]):
            if not isinstance(item, dict):
                continue
            expression = str(item.get("expression", "")).strip()
            if not expression:
                continue
            source_pair = item.get("source_pair", [])
            if not isinstance(source_pair, list) or len(source_pair) < 2:
                fallback = top_pairs[idx] if idx < len(top_pairs) else None
                if fallback is not None:
                    source_pair = [fallback["left_id"], fallback["right_id"]]
                else:
                    source_pair = []
            novelty_score = _expression_novelty(expression, reference_expressions)
            llm_is_new = item.get("is_new_theory")
            if isinstance(llm_is_new, bool):
                is_new_theory = bool(llm_is_new) and novelty_score >= 0.2
            else:
                is_new_theory = novelty_score >= 0.35

            synthesized.append(
                {
                    "title": str(item.get("title", f"collision_candidate_{idx + 1}"))[:160],
                    "rule_text": str(item.get("rule_text", ""))[:400],
                    "expression": expression,
                    "rationale": str(item.get("rationale", ""))[:1200],
                    "confidence": _coerce_confidence(item.get("confidence", 0.5)),
                    "source_pair": source_pair[:2],
                    "is_new_theory": is_new_theory,
                    "novelty_score": novelty_score,
                    "novelty_reason": str(item.get("novelty_reason", ""))[:400],
                }
            )

        return {
            "provider": settings.normalized_provider(),
            "model": settings.model_name,
            "pair_scores": top_pairs,
            "collision_hypotheses": synthesized,
            "new_theory_count": len([item for item in synthesized if item["is_new_theory"]]),
            "reference_expressions_count": len(reference_expressions),
        }

    @app.get("/api/readme")
    def readme() -> dict:
        readme_path = repo_root / "README.md"
        if not readme_path.exists():
            raise HTTPException(status_code=404, detail="README.md not found")
        return {
            "path": str(readme_path),
            "content": readme_path.read_text(encoding="utf-8"),
        }

    @app.get("/api/configs")
    def configs() -> dict:
        quick = repo_root / "config" / "quickstart.yaml"
        full = repo_root / "config" / "full_experiment.yaml"
        openworld = repo_root / "config" / "openworld_pathfinder.yaml"
        return {
            "quickstart": quick.read_text(encoding="utf-8") if quick.exists() else "",
            "full_experiment": full.read_text(encoding="utf-8") if full.exists() else "",
            "openworld_pathfinder": openworld.read_text(encoding="utf-8") if openworld.exists() else "",
        }

    @app.post("/api/provider-models")
    def provider_models(req: ProviderModelsRequest) -> dict:
        settings = _build_runtime_settings(req)
        provider_error = settings.validate_provider_config()
        if provider_error and "required when MODEL_PROVIDER" not in provider_error:
            raise HTTPException(status_code=400, detail=provider_error)

        try:
            models, source_endpoint = _fetch_available_models(settings)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            detail = (
                f"Model list endpoint returned {exc.response.status_code}: "
                f"{exc.response.text[:220]}"
            )
            raise HTTPException(status_code=502, detail=detail) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch provider models: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Failed to fetch provider models: {exc}") from exc

        models, reasoner_models = _split_reasoner_models(models)
        return {
            "provider": settings.normalized_provider(),
            "source_endpoint": source_endpoint,
            "models": models,
            "reasoner_models": reasoner_models,
            "count": len(models),
            "reasoner_count": len(reasoner_models),
            "fetched_at": utc_now(),
        }

    @app.post("/api/run")
    def start_run(req: RunRequest) -> dict:
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        now = utc_now()
        job = RunJob(
            job_id=job_id,
            status="queued",
            created_at=now,
            updated_at=now,
            request=req.model_dump(),
        )
        with jobs_lock:
            jobs[job_id] = job
            job_cancel_flags[job_id] = threading.Event()
            _persist_jobs_locked()
        _append_job_log(job_id, "Job queued")

        thread = threading.Thread(target=_execute_run_job, args=(job_id, req), daemon=True)
        thread.start()
        return job.model_dump()

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict:
        should_log_stop_request = False
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

            if job.status in {"completed", "failed", "cancelled"}:
                raise HTTPException(status_code=409, detail=f"Job cannot be stopped from status: {job.status}")

            flag = job_cancel_flags.get(job_id)
            if flag is None:
                flag = threading.Event()
                job_cancel_flags[job_id] = flag

            if not flag.is_set():
                flag.set()
                should_log_stop_request = True

            payload = job.model_dump()
            if payload.get("status") in {"queued", "running"}:
                payload["status"] = "stopping"
                payload["updated_at"] = utc_now()
                jobs[job_id] = RunJob.model_validate(payload)
                _persist_jobs_locked()

        if should_log_stop_request:
            _append_job_log(job_id, "Stop requested by user", level="warning")

        with jobs_lock:
            return jobs[job_id].model_dump()

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        with jobs_lock:
            ordered = sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [job.model_dump() for job in ordered]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        with jobs_lock:
            if job_id not in jobs:
                raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
            return jobs[job_id].model_dump()

    return app
