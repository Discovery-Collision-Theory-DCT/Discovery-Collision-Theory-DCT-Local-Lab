import csv
from pathlib import Path

from dct.config import AblationConfig, ExperimentConfig, RuntimeSettings
from dct.memory import SQLiteMemory
from dct.orchestration import DCTOrchestrator
from tests.fake_provider import FakeProvider


def test_openworld_metrics_are_emitted(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DCT_SQLITE_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("DCT_OUTPUT_DIR", str(tmp_path / "outputs"))

    settings = RuntimeSettings()
    memory = SQLiteMemory(settings.dct_sqlite_path)

    config = ExperimentConfig(
        name="openworld_metrics",
        seed=7,
        trials=1,
        rounds=1,
        hypotheses_per_trajectory=1,
        baselines=["full_dct"],
        benchmark_families=["open_world_noise"],
        samples_per_task_train=12,
        samples_per_task_heldout=8,
        ablation=AblationConfig(),
        output_dir=tmp_path / "outputs",
    )

    orchestrator = DCTOrchestrator(settings=settings, provider=FakeProvider(), memory=memory)
    summary, out_dir = orchestrator.run(config)
    memory.close()

    assert summary.method_summaries
    method = summary.method_summaries[0]
    assert method.ood_predictive_accuracy >= 0.0
    assert method.stress_predictive_accuracy >= 0.0
    assert method.transfer_generalization_score >= 0.0
    assert method.open_world_readiness_score >= 0.0

    csv_path = out_dir / "method_summaries.csv"
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
    assert "ood_predictive_accuracy" in headers
    assert "stress_predictive_accuracy" in headers
    assert "transfer_generalization_score" in headers
    assert "open_world_readiness_score" in headers
