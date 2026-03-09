import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from dct.api.app import create_app
from dct.config import ExperimentConfig
from dct.orchestration import RunCancelledError


class _DummyProvider:
    def check_health(self):
        return True, "ok"

    def generate_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 800):
        if "collision_hypotheses" in system_prompt:
            return {
                "collision_hypotheses": [
                    {
                        "title": "Synthesized theory",
                        "rule_text": "Combined mechanism",
                        "expression": "x + y",
                        "rationale": "Combined from two directions",
                        "confidence": 0.88,
                        "source_pair": ["d1", "d2"],
                        "is_new_theory": True,
                        "novelty_reason": "Low overlap with prior expressions",
                    }
                ]
            }
        return {
            "executive_summary": "full_dct shows stronger multi-metric behavior",
            "key_findings": ["validity and heldout accuracy improved"],
            "risks": ["limited to controlled benchmarks"],
            "recommended_next_experiments": ["run more trials with OOD stress emphasis"],
            "confidence": 0.72,
        }

    def set_debug_callback(self, callback):
        return


def _write_demo_summary(tmp_path: Path, run_name: str = "demo_run") -> None:
    run_dir = tmp_path / "quickstart" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": run_name,
        "config": {"name": "quickstart"},
        "method_summaries": [
            {
                "method": "full_dct",
                "trial_index": 0,
                "validity_rate": 0.6,
                "heldout_predictive_accuracy": 0.7,
                "ood_predictive_accuracy": 0.65,
                "stress_predictive_accuracy": 0.61,
                "transfer_generalization_score": 0.68,
                "open_world_readiness_score": 0.66,
                "rule_recovery_exact_match_rate": 0.31,
                "compression_score": 0.04,
                "novelty_score": 0.81,
                "time_to_valid_discovery": 1.0,
                "cumulative_improvement": 0.32,
                "rounds": [],
            }
        ],
        "uplift": {"baseline_single_a": {"heldout_predictive_accuracy": 0.1}},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_explain_requires_online_remote(tmp_path: Path):
    _write_demo_summary(tmp_path)
    app = create_app(tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/api/runs/demo_run/explain",
        json={
            "model_provider": "openai_compatible",
            "model_access_mode": "local",
            "openai_base_url": "http://localhost:11434/v1",
        },
    )
    assert resp.status_code == 400
    assert "online remote providers" in resp.text


def test_explain_online_with_mock_provider(monkeypatch, tmp_path: Path):
    _write_demo_summary(tmp_path)
    app = create_app(tmp_path)
    client = TestClient(app)
    monkeypatch.setattr("dct.api.app.build_provider", lambda settings: _DummyProvider())

    resp = client.post(
        "/api/runs/demo_run/explain",
        json={
            "model_provider": "openai",
            "model_access_mode": "online",
            "allow_remote_inference": True,
            "openai_base_url": "https://api.openai.com/v1",
            "openai_api_key": "sk-test",
            "model_name": "gpt-test",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_name"] == "demo_run"
    assert body["provider"] == "openai"
    assert "executive_summary" in body["explanation"]


def test_discovery_collide_with_mock_provider(monkeypatch, tmp_path: Path):
    app = create_app(tmp_path)
    client = TestClient(app)
    monkeypatch.setattr("dct.api.app.build_provider", lambda settings: _DummyProvider())

    resp = client.post(
        "/api/discovery/collide",
        json={
            "model_provider": "openai_compatible",
            "model_access_mode": "local",
            "openai_base_url": "http://localhost:11434/v1",
            "discoveries": [
                {
                    "discovery_id": "d1",
                    "title": "A",
                    "expression": "x + 1",
                    "direction": {"x": 1, "y": 0, "z": 0},
                    "confidence": 0.7,
                },
                {
                    "discovery_id": "d2",
                    "title": "B",
                    "expression": "y - 1",
                    "direction": {"x": -1, "y": 0.2, "z": 0},
                    "confidence": 0.8,
                },
            ],
            "known_theories": ["x + y + 3"],
            "memory_expressions": ["x - y"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["collision_hypotheses"]) == 1
    assert "new_theory_count" in body


def test_jobs_are_persisted_to_disk(tmp_path: Path):
    app = create_app(tmp_path)
    client = TestClient(app)

    run_resp = client.post(
        "/api/run",
        json={
            "mode": "quickstart",
            "config_path": str(tmp_path / "missing_config.yaml"),
            "skip_model_check": True,
            "model_provider": "openai_compatible",
            "model_access_mode": "local",
            "openai_base_url": "http://localhost:11434/v1",
        },
    )
    assert run_resp.status_code == 200
    job_id = run_resp.json()["job_id"]

    final = None
    for _ in range(40):
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"failed", "completed"}:
            final = payload
            break
        time.sleep(0.05)
    assert final is not None

    app2 = create_app(tmp_path)
    client2 = TestClient(app2)
    jobs = client2.get("/api/jobs").json()
    ids = {job["job_id"] for job in jobs}
    assert job_id in ids


def test_stop_job_cancels_running_job(monkeypatch, tmp_path: Path):
    class _SlowOrchestrator:
        def __init__(self, settings, provider, memory):
            self.settings = settings
            self.provider = provider
            self.memory = memory

        def run(self, exp_config, progress_callback=None, should_stop=None):
            if progress_callback is not None:
                progress_callback(
                    {
                        "type": "run_started",
                        "run_name": "slow_run",
                        "baselines": ["baseline_single_a"],
                        "trials": 1,
                        "rounds": 1,
                    }
                )
            for i in range(120):
                time.sleep(0.01)
                if callable(should_stop) and should_stop():
                    raise RunCancelledError("cancelled in test slow orchestrator")
                if progress_callback is not None:
                    progress_callback(
                        {
                            "type": "task_started",
                            "method": "baseline_single_a",
                            "trial_index": 0,
                            "round_index": i,
                            "family": "symbolic",
                        }
                    )

            class _Summary:
                run_name = "slow_run"

            return _Summary(), Path(exp_config.output_dir) / "slow_run"

    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("name: dummy\n", encoding="utf-8")

    monkeypatch.setattr("dct.api.app.build_provider", lambda settings: _DummyProvider())
    monkeypatch.setattr("dct.api.app.load_experiment_config", lambda path: ExperimentConfig(output_dir=tmp_path / "runs"))
    monkeypatch.setattr("dct.api.app.DCTOrchestrator", _SlowOrchestrator)

    app = create_app(tmp_path)
    client = TestClient(app)

    run_resp = client.post(
        "/api/run",
        json={
            "mode": "quickstart",
            "config_path": str(config_path),
            "skip_model_check": True,
            "model_provider": "openai_compatible",
            "model_access_mode": "local",
            "openai_base_url": "http://localhost:11434/v1",
        },
    )
    assert run_resp.status_code == 200
    job_id = run_resp.json()["job_id"]

    stop_resp = client.post(f"/api/jobs/{job_id}/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] in {"stopping", "cancelled"}

    final = None
    for _ in range(80):
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"cancelled", "failed", "completed"}:
            final = payload
            break
        time.sleep(0.05)

    assert final is not None
    assert final["status"] == "cancelled"
    assert final.get("error") in (None, "")
    assert any("Stop requested by user" in str(item.get("message", "")) for item in final.get("logs", []))


def test_provider_models_endpoint_returns_model_and_reasoner_lists(monkeypatch, tmp_path: Path):
    app = create_app(tmp_path)
    client = TestClient(app)

    monkeypatch.setattr(
        "dct.api.app._fetch_available_models",
        lambda settings: (["gpt-5-mini", "deepseek-reasoner", "o3"], "https://api.openai.com/v1/models"),
    )

    resp = client.post(
        "/api/provider-models",
        json={
            "model_provider": "openai",
            "model_access_mode": "online",
            "allow_remote_inference": True,
            "openai_base_url": "https://api.openai.com/v1",
            "openai_api_key": "sk-test",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "openai"
    assert "gpt-5-mini" in body["models"]
    assert "deepseek-reasoner" in body["reasoner_models"]
    assert "o3" in body["reasoner_models"]


def test_provider_models_endpoint_missing_anthropic_key(tmp_path: Path):
    app = create_app(tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/api/provider-models",
        json={
            "model_provider": "anthropic",
            "anthropic_base_url": "https://api.anthropic.com",
        },
    )
    assert resp.status_code == 400
    assert "ANTHROPIC_API_KEY" in resp.text
