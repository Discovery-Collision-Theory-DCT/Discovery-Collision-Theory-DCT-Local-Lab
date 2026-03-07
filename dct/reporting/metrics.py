from __future__ import annotations

from statistics import mean

from dct.schemas import CandidateLogRecord, MethodSummary, RoundSummary


def compute_method_summary(
    method: str,
    trial_index: int,
    round_summaries: list[RoundSummary],
    candidate_logs: list[CandidateLogRecord],
) -> MethodSummary:
    total_candidates = len(candidate_logs)
    accepted = [c for c in candidate_logs if c.accepted]

    validity_rate = (len(accepted) / total_candidates) if total_candidates else 0.0

    if round_summaries:
        heldout_predictive_accuracy = mean([r.top_heldout_accuracy for r in round_summaries])
    else:
        heldout_predictive_accuracy = 0.0

    if accepted:
        exact_rate = mean([1.0 if c.exact_match else 0.0 for c in accepted])
    else:
        exact_rate = 0.0

    compression_values = [1.0 / (1.0 + len(c.expression)) for c in candidate_logs if c.expression]
    compression_score = mean(compression_values) if compression_values else 0.0

    novelty_score = mean([c.novelty for c in candidate_logs]) if candidate_logs else 0.0

    valid_rounds = [r.round_index for r in round_summaries if r.accepted_count > 0]
    time_to_valid = float(min(valid_rounds)) if valid_rounds else float(len(round_summaries) + 1)

    cumulative = 0.0
    running_best = 0.0
    for r in sorted(round_summaries, key=lambda x: x.round_index):
        if r.top_heldout_accuracy > running_best:
            cumulative += r.top_heldout_accuracy - running_best
            running_best = r.top_heldout_accuracy

    return MethodSummary(
        method=method,
        trial_index=trial_index,
        validity_rate=float(validity_rate),
        heldout_predictive_accuracy=float(heldout_predictive_accuracy),
        rule_recovery_exact_match_rate=float(exact_rate),
        compression_score=float(compression_score),
        novelty_score=float(novelty_score),
        time_to_valid_discovery=float(time_to_valid),
        cumulative_improvement=float(cumulative),
        rounds=round_summaries,
    )


def compute_uplift(method_summaries: list[MethodSummary]) -> dict[str, dict[str, float]]:
    by_method: dict[str, list[MethodSummary]] = {}
    for item in method_summaries:
        by_method.setdefault(item.method, []).append(item)

    def mean_metric(method: str, metric: str) -> float:
        vals = [getattr(m, metric) for m in by_method.get(method, [])]
        return float(mean(vals)) if vals else 0.0

    full_name = "full_dct"
    full_metrics = {
        "validity_rate": mean_metric(full_name, "validity_rate"),
        "heldout_predictive_accuracy": mean_metric(full_name, "heldout_predictive_accuracy"),
        "rule_recovery_exact_match_rate": mean_metric(full_name, "rule_recovery_exact_match_rate"),
        "cumulative_improvement": mean_metric(full_name, "cumulative_improvement"),
    }

    uplift: dict[str, dict[str, float]] = {}
    for baseline in ["baseline_single_a", "baseline_single_b", "baseline_merged_naive"]:
        uplift[baseline] = {}
        for metric, full_val in full_metrics.items():
            base_val = mean_metric(baseline, metric)
            uplift[baseline][metric] = full_val - base_val
    return uplift
