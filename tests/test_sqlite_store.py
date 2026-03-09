import json

from dct.memory import SQLiteMemory
from dct.schemas import RobustnessGateResult, VerificationResult, VerifierMode, VerifierVerdict


def test_log_verdict_persists_robustness_gate_row(tmp_path):
    memory = SQLiteMemory(tmp_path / "memory.db")
    verdict = VerifierVerdict(
        hypothesis_id="hyp-1",
        passed=True,
        confidence=0.91,
        reason="ok",
        per_mode=[
            VerificationResult(
                hypothesis_id="hyp-1",
                mode=VerifierMode.predictive,
                passed=True,
                confidence=0.93,
                reason="heldout pass",
                metrics={"heldout_accuracy": 0.93, "n": 10.0},
            )
        ],
        robustness=RobustnessGateResult(
            enabled=True,
            passed=True,
            confidence=0.8,
            reason="ood/stress pass",
            ood_passed=True,
            stress_passed=True,
            ood_checked=True,
            stress_checked=True,
            metrics={
                "ood_accuracy": 0.8,
                "ood_n": 10.0,
                "stress_accuracy": 0.8,
                "stress_n": 10.0,
                "ood_threshold": 0.7,
                "stress_threshold": 0.65,
            },
        ),
    )

    memory.log_verdict(run_id="run-1", round_index=0, verdict=verdict)
    rows = memory.conn.execute(
        "SELECT mode, passed, confidence, reason, metrics_json FROM verifications ORDER BY id"
    ).fetchall()
    memory.close()

    assert len(rows) == 2
    mode_rows = {row["mode"]: row for row in rows}
    assert "predictive" in mode_rows
    assert "robustness_gate" in mode_rows
    robustness_metrics = json.loads(mode_rows["robustness_gate"]["metrics_json"])
    assert robustness_metrics["ood_accuracy"] == 0.8
    assert robustness_metrics["stress_accuracy"] == 0.8
    assert robustness_metrics["ood_checked"] == 1.0
    assert robustness_metrics["stress_checked"] == 1.0


def test_log_verdict_skips_disabled_robustness_gate_row(tmp_path):
    memory = SQLiteMemory(tmp_path / "memory.db")
    verdict = VerifierVerdict(
        hypothesis_id="hyp-2",
        passed=True,
        confidence=0.88,
        reason="ok",
        per_mode=[
            VerificationResult(
                hypothesis_id="hyp-2",
                mode=VerifierMode.predictive,
                passed=True,
                confidence=0.88,
                reason="heldout pass",
                metrics={"heldout_accuracy": 0.88, "n": 10.0},
            )
        ],
        robustness=RobustnessGateResult(
            enabled=False,
            passed=True,
            confidence=None,
            reason="disabled_for_single_mode_ablation",
            ood_passed=True,
            stress_passed=True,
            ood_checked=False,
            stress_checked=False,
            metrics={},
        ),
    )

    memory.log_verdict(run_id="run-2", round_index=0, verdict=verdict)
    rows = memory.conn.execute("SELECT mode FROM verifications ORDER BY id").fetchall()
    memory.close()

    assert len(rows) == 1
    assert rows[0]["mode"] == "predictive"
