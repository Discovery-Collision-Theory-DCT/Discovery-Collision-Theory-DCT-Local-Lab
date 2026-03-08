from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from dct.schemas import BenchmarkTask, Hypothesis, VerifierVerdict


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteMemory:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              method TEXT NOT NULL,
              trial_index INTEGER NOT NULL,
              config_json TEXT NOT NULL,
              started_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              round_index INTEGER NOT NULL,
              family TEXT NOT NULL,
              task_id TEXT NOT NULL,
              split TEXT NOT NULL,
              features_json TEXT NOT NULL,
              target_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hypotheses (
              hypothesis_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              round_index INTEGER NOT NULL,
              family TEXT NOT NULL,
              task_id TEXT NOT NULL,
              source TEXT NOT NULL,
              expression TEXT NOT NULL,
              rule_text TEXT NOT NULL,
              rationale TEXT NOT NULL,
              confidence REAL NOT NULL,
              parents_json TEXT NOT NULL,
              scores_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS verifications (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              round_index INTEGER NOT NULL,
              hypothesis_id TEXT NOT NULL,
              mode TEXT NOT NULL,
              passed INTEGER NOT NULL,
              confidence REAL NOT NULL,
              reason TEXT NOT NULL,
              metrics_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS accepted_theories (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              round_index INTEGER NOT NULL,
              family TEXT NOT NULL,
              task_id TEXT NOT NULL,
              hypothesis_id TEXT NOT NULL,
              predictive_accuracy REAL NOT NULL,
              improvement_delta REAL NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lineage (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              child_hypothesis_id TEXT NOT NULL,
              parent_hypothesis_id TEXT NOT NULL,
              relation TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def create_run(self, run_id: str, method: str, trial_index: int, config: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs(run_id, method, trial_index, config_json, started_at) VALUES(?,?,?,?,?)",
            (run_id, method, trial_index, json.dumps(config), utc_now()),
        )
        self.conn.commit()

    def log_observations(self, run_id: str, round_index: int, task: BenchmarkTask) -> None:
        rows = []
        for ex in task.train:
            rows.append(
                (
                    run_id,
                    round_index,
                    task.family,
                    task.task_id,
                    "train",
                    json.dumps(ex.features),
                    json.dumps(ex.target),
                )
            )
        for ex in task.heldout:
            rows.append(
                (
                    run_id,
                    round_index,
                    task.family,
                    task.task_id,
                    "heldout",
                    json.dumps(ex.features),
                    json.dumps(ex.target),
                )
            )
        for ex in task.ood:
            rows.append(
                (
                    run_id,
                    round_index,
                    task.family,
                    task.task_id,
                    "ood",
                    json.dumps(ex.features),
                    json.dumps(ex.target),
                )
            )
        for ex in task.stress:
            rows.append(
                (
                    run_id,
                    round_index,
                    task.family,
                    task.task_id,
                    "stress",
                    json.dumps(ex.features),
                    json.dumps(ex.target),
                )
            )
        self.conn.executemany(
            """
            INSERT INTO observations(run_id, round_index, family, task_id, split, features_json, target_json)
            VALUES(?,?,?,?,?,?,?)
            """,
            rows,
        )
        self.conn.commit()

    def log_hypothesis(self, run_id: str, hypothesis: Hypothesis) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO hypotheses(
                hypothesis_id, run_id, round_index, family, task_id, source,
                expression, rule_text, rationale, confidence, parents_json, scores_json, created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                hypothesis.hypothesis_id,
                run_id,
                hypothesis.round_index,
                hypothesis.family,
                hypothesis.task_id,
                hypothesis.source,
                hypothesis.expression,
                hypothesis.rule_text,
                hypothesis.rationale,
                float(hypothesis.confidence),
                json.dumps(hypothesis.parents),
                json.dumps(hypothesis.scores),
                utc_now(),
            ),
        )
        for parent in hypothesis.parents:
            self.conn.execute(
                "INSERT INTO lineage(run_id, child_hypothesis_id, parent_hypothesis_id, relation) VALUES(?,?,?,?)",
                (run_id, hypothesis.hypothesis_id, parent, "collision_parent"),
            )
        self.conn.commit()

    def log_verdict(self, run_id: str, round_index: int, verdict: VerifierVerdict) -> None:
        for mode_result in verdict.per_mode:
            self.conn.execute(
                """
                INSERT INTO verifications(
                    run_id, round_index, hypothesis_id, mode, passed, confidence, reason, metrics_json, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    round_index,
                    verdict.hypothesis_id,
                    mode_result.mode.value,
                    int(mode_result.passed),
                    float(mode_result.confidence),
                    mode_result.reason,
                    json.dumps(mode_result.metrics),
                    utc_now(),
                ),
            )
        self.conn.commit()

    def get_recent_theory_summaries(self, run_id: str, limit: int = 12) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT h.expression, h.rule_text, a.round_index
            FROM accepted_theories a
            JOIN hypotheses h ON a.hypothesis_id = h.hypothesis_id
            WHERE a.run_id = ?
            ORDER BY a.id DESC
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        return [f"round={r['round_index']} expr={r['expression']} rule={r['rule_text']}" for r in rows]

    def get_recent_expressions(self, run_id: str, limit: int = 20) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT h.expression
            FROM accepted_theories a
            JOIN hypotheses h ON a.hypothesis_id = h.hypothesis_id
            WHERE a.run_id = ?
            ORDER BY a.id DESC
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        return [r["expression"] for r in rows]

    def get_best_accuracy_so_far(self, run_id: str, family: str) -> float:
        row = self.conn.execute(
            """
            SELECT MAX(predictive_accuracy) AS best_acc
            FROM accepted_theories
            WHERE run_id = ? AND family = ?
            """,
            (run_id, family),
        ).fetchone()
        if not row or row["best_acc"] is None:
            return 0.0
        return float(row["best_acc"])

    def log_accepted(
        self,
        run_id: str,
        round_index: int,
        family: str,
        task_id: str,
        hypothesis_id: str,
        predictive_accuracy: float,
    ) -> float:
        best_before = self.get_best_accuracy_so_far(run_id, family)
        improvement_delta = float(predictive_accuracy) - float(best_before)
        self.conn.execute(
            """
            INSERT INTO accepted_theories(
                run_id, round_index, family, task_id, hypothesis_id,
                predictive_accuracy, improvement_delta, created_at
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                round_index,
                family,
                task_id,
                hypothesis_id,
                float(predictive_accuracy),
                improvement_delta,
                utc_now(),
            ),
        )
        self.conn.commit()
        return improvement_delta

    def close(self) -> None:
        self.conn.close()
