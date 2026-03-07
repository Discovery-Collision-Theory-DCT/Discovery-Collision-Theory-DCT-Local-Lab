from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from dct.config import RuntimeSettings, load_experiment_config
from dct.llm import build_provider
from dct.memory import SQLiteMemory
from dct.orchestration import DCTOrchestrator



def main() -> None:
    load_dotenv()
    settings = RuntimeSettings()
    config = load_experiment_config(Path("config/quickstart.yaml"))

    provider = build_provider(settings)
    ok, msg = provider.check_health()
    if not ok:
        raise RuntimeError(f"Model endpoint unavailable: {msg}")

    memory = SQLiteMemory(settings.dct_sqlite_path)
    try:
        orchestrator = DCTOrchestrator(settings=settings, provider=provider, memory=memory)
        summary, out_dir = orchestrator.run(config)
        print(f"Run complete: {summary.run_name}")
        print(f"Artifacts: {out_dir}")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
