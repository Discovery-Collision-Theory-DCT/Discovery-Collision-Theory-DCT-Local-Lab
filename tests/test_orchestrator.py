from pathlib import Path

from dct.config import AblationConfig, ExperimentConfig, RuntimeSettings
from dct.memory import SQLiteMemory
from dct.orchestration import DCTOrchestrator
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
