from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from dct.agents.collision_engine import CollisionEngine
from dct.agents.trajectory_a import TrajectoryAAgent
from dct.agents.trajectory_b import TrajectoryBAgent
from dct.agents.verifier import Verifier
from dct.benchmarks import BenchmarkRegistry
from dct.config import ExperimentConfig, RuntimeSettings
from dct.memory import SQLiteMemory
from dct.orchestration.baselines import (
    BASELINE_MERGED_NAIVE,
    BASELINE_SINGLE_A,
    BASELINE_SINGLE_B,
    FULL_DCT,
)
from dct.reporting import ResultWriter, compute_method_summary, compute_uplift, generate_plots
from dct.schemas import (
    CandidateLogRecord,
    ExperimentSummary,
    Hypothesis,
    RoundSummary,
    VerifierMode,
)
from dct.utils import jaccard_similarity, new_id, normalize_expr, safe_eval_expression, token_set


class DCTOrchestrator:
    def __init__(self, settings: RuntimeSettings, provider, memory: SQLiteMemory):
        self.settings = settings
        self.provider = provider
        self.memory = memory
        self.registry = BenchmarkRegistry()

        self.trajectory_a = TrajectoryAAgent(provider)
        self.trajectory_b = TrajectoryBAgent(provider)
        self.collision_engine = CollisionEngine(provider)
        self.verifier = Verifier(provider)

    def run(
        self,
        config: ExperimentConfig,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[ExperimentSummary, Path]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.name}_{timestamp}"
        run_output_dir = Path(config.output_dir) / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        self._emit(
            progress_callback,
            {
                "type": "run_started",
                "run_name": run_name,
                "config_name": config.name,
                "baselines": config.baselines,
                "trials": config.trials,
                "rounds": config.rounds,
            },
        )

        method_summaries = []
        candidate_logs: list[CandidateLogRecord] = []

        for method in config.baselines:
            for trial_index in range(config.trials):
                self._emit(
                    progress_callback,
                    {
                        "type": "trial_started",
                        "run_name": run_name,
                        "method": method,
                        "trial_index": trial_index,
                    },
                )
                run_id = f"{run_name}_{method}_trial{trial_index}"
                self.memory.create_run(
                    run_id=run_id,
                    method=method,
                    trial_index=trial_index,
                    config=config.model_dump(mode="json"),
                )
                round_summaries, trial_logs = self._run_single_method_trial(
                    run_id=run_id,
                    method=method,
                    trial_index=trial_index,
                    config=config,
                    progress_callback=progress_callback,
                )
                candidate_logs.extend(trial_logs)
                accepted_count = len([r for r in trial_logs if r.accepted])
                method_summaries.append(
                    compute_method_summary(
                        method=method,
                        trial_index=trial_index,
                        round_summaries=round_summaries,
                        candidate_logs=trial_logs,
                    )
                )
                self._emit(
                    progress_callback,
                    {
                        "type": "trial_completed",
                        "run_name": run_name,
                        "method": method,
                        "trial_index": trial_index,
                        "candidate_count": len(trial_logs),
                        "accepted_count": accepted_count,
                    },
                )

        summary = ExperimentSummary(
            run_name=run_name,
            config=config.model_dump(mode="json"),
            method_summaries=method_summaries,
            uplift=compute_uplift(method_summaries),
        )

        writer = ResultWriter(run_output_dir)
        writer.write_candidate_logs(candidate_logs)
        writer.write_method_summaries(method_summaries)
        writer.write_round_jsonl(method_summaries)
        writer.write_summary_json(summary)
        generate_plots(method_summaries, candidate_logs, run_output_dir)

        self._emit(
            progress_callback,
            {
                "type": "run_completed",
                "run_name": run_name,
                "run_output_dir": str(run_output_dir),
                "method_summary_count": len(method_summaries),
            },
        )

        return summary, run_output_dir

    def _run_single_method_trial(
        self,
        run_id: str,
        method: str,
        trial_index: int,
        config: ExperimentConfig,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[RoundSummary], list[CandidateLogRecord]]:
        round_summaries: list[RoundSummary] = []
        candidate_logs: list[CandidateLogRecord] = []
        first_valid_round: int | None = None

        verifier_modes = config.verifier_modes
        if config.ablation.single_verifier_mode_only:
            verifier_modes = [config.ablation.single_verifier_mode_only]

        for round_index in range(config.rounds):
            for family_index, family in enumerate(config.benchmark_families):
                task_seed = config.seed + (trial_index * 10000) + (round_index * 100) + family_index
                self._emit(
                    progress_callback,
                    {
                        "type": "task_started",
                        "run_id": run_id,
                        "method": method,
                        "trial_index": trial_index,
                        "round_index": round_index,
                        "family": family,
                        "task_seed": task_seed,
                    },
                )
                task = self.registry.generate(
                    family=family,
                    seed=task_seed,
                    n_train=config.samples_per_task_train,
                    n_heldout=config.samples_per_task_heldout,
                )
                self.memory.log_observations(run_id, round_index, task)

                memory_summaries = []
                memory_expressions = []
                if not config.ablation.no_memory_write_back:
                    memory_summaries = self.memory.get_recent_theory_summaries(run_id)
                    memory_expressions = self.memory.get_recent_expressions(run_id)

                hyp_a: list[Hypothesis] = []
                hyp_b: list[Hypothesis] = []

                if method in {BASELINE_SINGLE_A, BASELINE_MERGED_NAIVE, FULL_DCT}:
                    hyp_a = self.trajectory_a.propose(
                        task=task,
                        round_index=round_index,
                        memory_summaries=memory_summaries,
                        hypotheses_to_generate=config.hypotheses_per_trajectory,
                    )

                if method in {BASELINE_SINGLE_B, BASELINE_MERGED_NAIVE, FULL_DCT}:
                    hyp_b = self.trajectory_b.propose(
                        task=task,
                        round_index=round_index,
                        memory_summaries=memory_summaries,
                        hypotheses_to_generate=config.hypotheses_per_trajectory,
                    )

                candidates = self._assemble_candidates(
                    method=method,
                    task=task,
                    round_index=round_index,
                    hyp_a=hyp_a,
                    hyp_b=hyp_b,
                    memory_expressions=memory_expressions,
                    no_collision=config.ablation.no_collision,
                )

                for c in candidates:
                    self.memory.log_hypothesis(run_id, c)

                round_records = self._verify_and_log_candidates(
                    run_id=run_id,
                    method=method,
                    trial_index=trial_index,
                    round_index=round_index,
                    task=task,
                    candidates=candidates,
                    verifier_modes=verifier_modes,
                    disable_verifier=config.ablation.no_verifier,
                    disable_memory_write=config.ablation.no_memory_write_back,
                    memory_expressions=memory_expressions,
                )

                candidate_logs.extend(round_records)
                accepted_records = [r for r in round_records if r.accepted]
                validity_rate = (len(accepted_records) / len(round_records)) if round_records else 0.0
                top_accuracy = (
                    max((r.predictive_accuracy for r in accepted_records), default=0.0)
                    if accepted_records
                    else max((r.predictive_accuracy for r in round_records), default=0.0)
                )
                avg_novelty = mean([r.novelty for r in round_records]) if round_records else 0.0

                if accepted_records and first_valid_round is None:
                    first_valid_round = round_index

                round_summaries.append(
                    RoundSummary(
                        round_index=round_index,
                        family=family,
                        task_id=task.task_id,
                        candidate_count=len(round_records),
                        accepted_count=len(accepted_records),
                        validity_rate=validity_rate,
                        top_heldout_accuracy=top_accuracy,
                        average_novelty=avg_novelty,
                        time_to_valid_discovery=first_valid_round,
                    )
                )
                self._emit(
                    progress_callback,
                    {
                        "type": "task_completed",
                        "run_id": run_id,
                        "method": method,
                        "trial_index": trial_index,
                        "round_index": round_index,
                        "family": family,
                        "task_id": task.task_id,
                        "candidate_count": len(round_records),
                        "accepted_count": len(accepted_records),
                        "validity_rate": validity_rate,
                        "top_heldout_accuracy": top_accuracy,
                        "average_novelty": avg_novelty,
                    },
                )

        return round_summaries, candidate_logs

    def _assemble_candidates(
        self,
        method: str,
        task,
        round_index: int,
        hyp_a: list[Hypothesis],
        hyp_b: list[Hypothesis],
        memory_expressions: list[str],
        no_collision: bool,
    ) -> list[Hypothesis]:
        if method == BASELINE_SINGLE_A:
            return hyp_a
        if method == BASELINE_SINGLE_B:
            return hyp_b
        if method == BASELINE_MERGED_NAIVE:
            return hyp_a + hyp_b

        if method == FULL_DCT:
            if no_collision:
                return hyp_a + hyp_b
            collision_hypotheses = self.collision_engine.collide(
                task=task,
                round_index=round_index,
                hypotheses_a=hyp_a,
                hypotheses_b=hyp_b,
                memory_expressions=memory_expressions,
                max_to_generate=max(1, min(len(hyp_a), len(hyp_b))),
            )
            return hyp_a + hyp_b + collision_hypotheses

        raise ValueError(f"Unknown method: {method}")

    def _verify_and_log_candidates(
        self,
        run_id: str,
        method: str,
        trial_index: int,
        round_index: int,
        task,
        candidates: list[Hypothesis],
        verifier_modes: list[str],
        disable_verifier: bool,
        disable_memory_write: bool,
        memory_expressions: list[str],
    ) -> list[CandidateLogRecord]:
        if not candidates:
            return []

        quick_metrics_cache: dict[str, tuple[float, float, float]] = {}
        for c in candidates:
            quick_metrics_cache[c.hypothesis_id] = self._quick_metrics(task, c.expression)

        if disable_verifier:
            sorted_candidates = sorted(
                candidates,
                key=lambda c: (quick_metrics_cache[c.hypothesis_id][0], c.confidence),
                reverse=True,
            )
            accepted_ids = {sorted_candidates[0].hypothesis_id}
        else:
            accepted_ids = set()

        records: list[CandidateLogRecord] = []

        for c in candidates:
            predictive_acc, symbolic_acc, simulation_acc = quick_metrics_cache[c.hypothesis_id]

            if disable_verifier:
                accepted = c.hypothesis_id in accepted_ids
            else:
                verdict = self.verifier.verify(task=task, hypothesis=c, modes=verifier_modes)
                self.memory.log_verdict(run_id=run_id, round_index=round_index, verdict=verdict)
                accepted = verdict.passed

            if accepted and not disable_memory_write:
                self.memory.log_accepted(
                    run_id=run_id,
                    round_index=round_index,
                    family=task.family,
                    task_id=task.task_id,
                    hypothesis_id=c.hypothesis_id,
                    predictive_accuracy=predictive_acc,
                )

            novelty = c.scores.get("novelty", self._novelty_from_memory(c.expression, memory_expressions))
            exact = normalize_expr(c.expression) == normalize_expr(task.ground_truth_expression)

            records.append(
                CandidateLogRecord(
                    run_id=run_id,
                    method=method,
                    trial_index=trial_index,
                    round_index=round_index,
                    family=task.family,
                    task_id=task.task_id,
                    hypothesis_id=c.hypothesis_id,
                    source=c.source,
                    expression=c.expression,
                    confidence=float(c.confidence),
                    novelty=float(novelty),
                    predictive_accuracy=float(predictive_acc),
                    symbolic_accuracy=float(symbolic_acc),
                    simulation_accuracy=float(simulation_acc),
                    accepted=bool(accepted),
                    exact_match=bool(exact),
                )
            )

        return records

    @staticmethod
    def _quick_metrics(task, expression: str) -> tuple[float, float, float]:
        predictive = DCTOrchestrator._accuracy(task.heldout, expression, float(task.metadata.get("target_tolerance", 0.0)))

        symbolic_tol = 0.0 if float(task.metadata.get("target_tolerance", 0.0)) == 0.0 else float(
            task.metadata.get("target_tolerance", 0.0)
        )
        symbolic = DCTOrchestrator._accuracy(task.train, expression, symbolic_tol)

        if task.simulation_cases:
            sim_examples = [
                type("Tmp", (), {"features": sc.features, "target": sc.expected_target}) for sc in task.simulation_cases
            ]
            simulation = DCTOrchestrator._accuracy(sim_examples, expression, float(task.metadata.get("target_tolerance", 0.0)))
        else:
            simulation = predictive

        return predictive, symbolic, simulation

    @staticmethod
    def _accuracy(examples, expression: str, tol: float) -> float:
        if not examples:
            return 0.0
        total = 0
        correct = 0
        for ex in examples:
            total += 1
            try:
                pred = safe_eval_expression(expression, ex.features)
            except ValueError:
                continue

            if isinstance(pred, (int, float)) and isinstance(ex.target, (int, float)):
                if abs(float(pred) - float(ex.target)) <= max(1e-9, tol):
                    correct += 1
            else:
                if pred == ex.target:
                    correct += 1
        if total == 0:
            return 0.0
        return correct / total

    @staticmethod
    def _novelty_from_memory(expression: str, memory_expressions: list[str]) -> float:
        if not memory_expressions:
            return 1.0
        expr_tokens = token_set(expression)
        sims = [jaccard_similarity(expr_tokens, token_set(mem_expr)) for mem_expr in memory_expressions]
        return max(0.0, min(1.0, 1.0 - max(sims)))

    @staticmethod
    def _emit(callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
        if callback is None:
            return
        try:
            callback(payload)
        except Exception:  # noqa: BLE001
            return
