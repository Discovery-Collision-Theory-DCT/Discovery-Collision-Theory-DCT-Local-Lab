from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dct.config import RuntimeSettings, load_experiment_config
from dct.llm import build_provider
from dct.memory import SQLiteMemory
from dct.orchestration import DCTOrchestrator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRequest(BaseModel):
    config_path: str | None = None
    mode: str = "quickstart"
    output_dir: str | None = None
    skip_model_check: bool = False

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


def create_app(output_root: Path) -> FastAPI:
    output_root = output_root.resolve()
    ui_root = Path(__file__).resolve().parents[1] / "ui"
    repo_root = Path(__file__).resolve().parents[2]

    app = FastAPI(title="DCT Local Dashboard API", version="0.2.0")
    app.mount("/outputs", StaticFiles(directory=str(output_root), check_dir=False), name="outputs")
    if ui_root.exists():
        app.mount("/ui-assets", StaticFiles(directory=str(ui_root), check_dir=True), name="ui-assets")

    jobs: dict[str, RunJob] = {}
    jobs_lock = threading.Lock()
    model_output_max_chars = 4000

    def _default_config_for_mode(mode: str) -> Path:
        if mode == "quickstart":
            return repo_root / "config" / "quickstart.yaml"
        if mode == "full":
            return repo_root / "config" / "full_experiment.yaml"
        raise ValueError(f"Unsupported mode: {mode}")

    def _build_runtime_settings(req: RunRequest) -> RuntimeSettings:
        overrides = {}
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
            job = jobs[job_id]
            payload = job.model_dump()
            payload.update(kwargs)
            payload["updated_at"] = utc_now()
            jobs[job_id] = RunJob.model_validate(payload)

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
            try:
                validity_str = f"{float(validity):.3f}"
            except (TypeError, ValueError):
                validity_str = "n/a"
            try:
                top_acc_str = f"{float(top_acc):.3f}"
            except (TypeError, ValueError):
                top_acc_str = "n/a"
            return (
                f"Task completed: method={event.get('method')} trial={event.get('trial_index')} "
                f"round={event.get('round_index')} family={event.get('family')} "
                f"candidates={event.get('candidate_count')} accepted={event.get('accepted_count')} "
                f"validity={validity_str} top_acc={top_acc_str}"
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
        _update_job(job_id, status="running")
        _append_job_log(job_id, "Job running")

        memory = None
        try:
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

            provider = build_provider(settings)
            if hasattr(provider, "set_debug_callback"):
                provider.set_debug_callback(lambda event: _append_model_output_log(job_id, event))
            if settings.dct_check_model_on_start and not req.skip_model_check:
                ok, message = provider.check_health()
                if not ok:
                    raise RuntimeError(f"Model check failed: {message}")
                _append_job_log(job_id, f"Model check passed: {message}")
            else:
                _append_job_log(job_id, "Model check skipped")

            config_path = Path(req.config_path).resolve() if req.config_path else _default_config_for_mode(req.mode)
            if not config_path.exists():
                raise RuntimeError(f"Config file not found: {config_path}")
            _append_job_log(job_id, f"Using config: {config_path}")

            exp_config = load_experiment_config(config_path)
            if req.output_dir:
                exp_config.output_dir = Path(req.output_dir)
            _append_job_log(job_id, f"Output dir: {exp_config.output_dir}")

            memory = SQLiteMemory(settings.dct_sqlite_path)
            orchestrator = DCTOrchestrator(settings=settings, provider=provider, memory=memory)
            summary, run_output_dir = orchestrator.run(
                exp_config,
                progress_callback=lambda event: _append_job_log(
                    job_id,
                    message=_event_to_log_message(event),
                    level="info",
                    event=event,
                ),
            )

            _update_job(
                job_id,
                status="completed",
                run_name=summary.run_name,
                run_output_dir=str(run_output_dir),
                error=None,
            )
            _append_job_log(job_id, f"Job completed: run_name={summary.run_name}")
        except Exception as exc:  # noqa: BLE001
            _update_job(job_id, status="failed", error=str(exc))
            _append_job_log(job_id, f"Job failed: {exc}", level="error")
        finally:
            if memory is not None:
                memory.close()

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
        return {
            "quickstart": quick.read_text(encoding="utf-8") if quick.exists() else "",
            "full_experiment": full.read_text(encoding="utf-8") if full.exists() else "",
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
        _append_job_log(job_id, "Job queued")

        thread = threading.Thread(target=_execute_run_job, args=(job_id, req), daemon=True)
        thread.start()
        return job.model_dump()

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
