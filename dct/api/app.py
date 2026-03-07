from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException


def create_app(output_root: Path) -> FastAPI:
    app = FastAPI(title="DCT Local Dashboard API", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/runs")
    def list_runs() -> list[dict]:
        runs = []
        if not output_root.exists():
            return runs

        for summary_file in sorted(output_root.glob("**/summary.json")):
            runs.append(
                {
                    "run_name": summary_file.parent.name,
                    "summary_path": str(summary_file),
                }
            )
        return runs

    @app.get("/runs/{run_name}")
    def get_run(run_name: str) -> dict:
        candidates = list(output_root.glob(f"**/{run_name}/summary.json"))
        if not candidates:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_name}")
        return json.loads(candidates[0].read_text(encoding="utf-8"))

    @app.get("/latest")
    def latest() -> dict:
        summaries = sorted(output_root.glob("**/summary.json"), key=lambda p: p.stat().st_mtime)
        if not summaries:
            raise HTTPException(status_code=404, detail="No experiment summaries found")
        return json.loads(summaries[-1].read_text(encoding="utf-8"))

    return app
