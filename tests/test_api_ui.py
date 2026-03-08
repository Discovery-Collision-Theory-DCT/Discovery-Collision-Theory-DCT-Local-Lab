import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from dct.api.app import create_app


def test_ui_and_run_listing(tmp_path: Path):
    run_dir = tmp_path / "quickstart" / "demo_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": "demo_run",
        "config": {},
        "method_summaries": [],
        "uplift": {},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "method_summaries.csv").write_text("method,validity_rate\n", encoding="utf-8")

    app = create_app(tmp_path)
    client = TestClient(app)

    index_resp = client.get("/")
    assert index_resp.status_code == 200
    assert "DCT Control Surface" in index_resp.text

    runs_resp = client.get("/api/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()
    assert runs
    assert runs[0]["run_name"] == "demo_run"

    run_resp = client.get("/api/runs/demo_run")
    assert run_resp.status_code == 200
    assert run_resp.json()["run_name"] == "demo_run"

    artifacts_resp = client.get("/api/runs/demo_run/artifacts")
    assert artifacts_resp.status_code == 200
    artifacts = artifacts_resp.json()["artifacts"]
    names = {item["name"] for item in artifacts}
    assert "summary.json" in names


def test_run_job_tracks_temperature_and_realtime_logs(tmp_path: Path):
    app = create_app(tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/api/run",
        json={
            "mode": "quickstart",
            "config_path": str(tmp_path / "missing_config.yaml"),
            "skip_model_check": True,
            "model_provider": "openai_compatible",
            "model_access_mode": "local",
            "model_name": "dummy-model",
            "model_temperature": 0.11,
        },
    )
    assert resp.status_code == 200
    job = resp.json()
    job_id = job["job_id"]

    final = None
    for _ in range(30):
        item = client.get(f"/api/jobs/{job_id}").json()
        if item["status"] in {"failed", "completed"}:
            final = item
            break
        time.sleep(0.05)

    assert final is not None
    assert final["request"]["model_temperature"] == 0.11
    assert isinstance(final.get("logs", []), list)
    assert len(final["logs"]) >= 1
