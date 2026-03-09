from pathlib import Path

from dct.config import AblationConfig, ExperimentConfig, RuntimeSettings
from dct.memory import SQLiteMemory
from dct.orchestration import DCTOrchestrator
from dct.schemas import Hypothesis
from dct.utils import new_id
from tests.fake_provider import FakeProvider


def test_orchestrator_runs_end_to_end_with_fake_provider(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DCT_SQLITE_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("DCT_OUTPUT_DIR", str(tmp_path / "outputs"))

    settings = RuntimeSettings()
    memory = SQLiteMemory(settings.dct_sqlite_path)

    config = ExperimentConfig(
        name="test_run",
        seed=42,
        trials=1,
        rounds=1,
        hypotheses_per_trajectory=1,
        baselines=["baseline_single_a", "baseline_single_b", "baseline_merged_naive", "full_dct"],
        benchmark_families=["symbolic"],
        samples_per_task_train=10,
        samples_per_task_heldout=6,
        verifier_modes=["predictive", "symbolic", "simulation"],
        ablation=AblationConfig(),
        output_dir=tmp_path / "outputs",
    )

    orchestrator = DCTOrchestrator(settings=settings, provider=FakeProvider(), memory=memory)
    summary, out_dir = orchestrator.run(config)
    memory.close()

    assert summary.method_summaries
    assert out_dir.exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "candidate_logs.csv").exists()


def test_single_verifier_mode_ablation_disables_robustness_gate(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DCT_SQLITE_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("DCT_OUTPUT_DIR", str(tmp_path / "outputs"))

    settings = RuntimeSettings()
    memory = SQLiteMemory(settings.dct_sqlite_path)

    config = ExperimentConfig(
        name="single_mode_ablation",
        seed=7,
        trials=1,
        rounds=1,
        hypotheses_per_trajectory=1,
        baselines=["baseline_single_a"],
        benchmark_families=["symbolic"],
        samples_per_task_train=8,
        samples_per_task_heldout=6,
        verifier_modes=["predictive", "symbolic", "simulation"],
        ablation=AblationConfig(single_verifier_mode_only="predictive"),
        output_dir=tmp_path / "outputs",
    )

    orchestrator = DCTOrchestrator(settings=settings, provider=FakeProvider(), memory=memory)
    seen_gate_flags: list[bool] = []
    original_verify = orchestrator.verifier.verify

    def _fixed_a_propose(task, round_index, memory_summaries, hypotheses_to_generate):
        feature = task.feature_names[0]
        return [
            Hypothesis(
                hypothesis_id=new_id("hypa"),
                source="trajectory_a",
                round_index=round_index,
                family=task.family,
                task_id=task.task_id,
                rule_text="fixed",
                expression=feature,
                rationale="fixed",
                confidence=0.8,
            )
        ]

    monkeypatch.setattr(orchestrator.trajectory_a, "propose", _fixed_a_propose)

    def _wrapped_verify(*args, **kwargs):
        seen_gate_flags.append(bool(kwargs.get("enable_robustness_gate", True)))
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(orchestrator.verifier, "verify", _wrapped_verify)
    orchestrator.run(config)
    memory.close()

    assert seen_gate_flags
    assert all(flag is False for flag in seen_gate_flags)
