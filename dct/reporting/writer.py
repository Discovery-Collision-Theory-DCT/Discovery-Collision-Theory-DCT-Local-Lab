from __future__ import annotations

import csv
import json
from pathlib import Path

from dct.schemas import CandidateLogRecord, ExperimentSummary, MethodSummary


class ResultWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_candidate_logs(self, records: list[CandidateLogRecord]) -> Path:
        out = self.output_dir / "candidate_logs.csv"
        if not records:
            out.write_text("", encoding="utf-8")
            return out

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].model_dump().keys()))
            writer.writeheader()
            for r in records:
                writer.writerow(r.model_dump())
        return out

    def write_method_summaries(self, method_summaries: list[MethodSummary]) -> Path:
        out = self.output_dir / "method_summaries.csv"
        rows: list[dict] = []
        for m in method_summaries:
            row = m.model_dump()
            row.pop("rounds", None)
            rows.append(row)

        if not rows:
            out.write_text("", encoding="utf-8")
            return out

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return out

    def write_summary_json(self, summary: ExperimentSummary) -> Path:
        out = self.output_dir / "summary.json"
        out.write_text(json.dumps(summary.model_dump(), indent=2), encoding="utf-8")
        return out

    def write_round_jsonl(self, method_summaries: list[MethodSummary]) -> Path:
        out = self.output_dir / "round_summaries.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for m in method_summaries:
                for round_summary in m.rounds:
                    payload = {
                        "method": m.method,
                        "trial_index": m.trial_index,
                        **round_summary.model_dump(),
                    }
                    f.write(json.dumps(payload) + "\n")
        return out
