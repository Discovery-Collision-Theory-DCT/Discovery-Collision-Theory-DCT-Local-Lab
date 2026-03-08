from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dct.schemas import CandidateLogRecord, MethodSummary


def generate_plots(method_summaries: list[MethodSummary], candidate_logs: list[CandidateLogRecord], output_dir: Path) -> list[Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    outputs.append(_plot_method_accuracy(method_summaries, plot_dir / "method_accuracy.png"))
    outputs.append(_plot_method_validity(method_summaries, plot_dir / "method_validity.png"))
    outputs.append(_plot_open_world_metrics(method_summaries, plot_dir / "open_world_metrics.png"))
    outputs.extend(_plot_family_accuracy(candidate_logs, plot_dir))
    outputs.append(_plot_cumulative_improvement(method_summaries, plot_dir / "cumulative_improvement.png"))
    return outputs


def _plot_method_accuracy(method_summaries: list[MethodSummary], out: Path) -> Path:
    methods = sorted({m.method for m in method_summaries})
    values = [
        mean([m.heldout_predictive_accuracy for m in method_summaries if m.method == method])
        for method in methods
    ]

    plt.figure(figsize=(8, 4))
    plt.bar(methods, values)
    plt.ylabel("Mean Held-out Accuracy")
    plt.title("Predictive Accuracy by Method")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def _plot_method_validity(method_summaries: list[MethodSummary], out: Path) -> Path:
    methods = sorted({m.method for m in method_summaries})
    values = [mean([m.validity_rate for m in method_summaries if m.method == method]) for method in methods]

    plt.figure(figsize=(8, 4))
    plt.bar(methods, values)
    plt.ylabel("Validity Rate")
    plt.title("Validity Rate by Method")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def _plot_family_accuracy(candidate_logs: list[CandidateLogRecord], plot_dir: Path) -> list[Path]:
    by_family_method: dict[tuple[str, str], list[float]] = defaultdict(list)
    families = sorted({c.family for c in candidate_logs})
    methods = sorted({c.method for c in candidate_logs})

    for c in candidate_logs:
        if c.accepted:
            by_family_method[(c.family, c.method)].append(c.predictive_accuracy)

    outputs: list[Path] = []
    for family in families:
        vals = []
        for method in methods:
            points = by_family_method.get((family, method), [])
            vals.append(mean(points) if points else 0.0)

        out = plot_dir / f"family_{family}_accuracy.png"
        plt.figure(figsize=(8, 4))
        plt.bar(methods, vals)
        plt.ylabel("Accepted Hypothesis Accuracy")
        plt.title(f"{family.title()} Benchmark Accuracy")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        outputs.append(out)

    return outputs


def _plot_cumulative_improvement(method_summaries: list[MethodSummary], out: Path) -> Path:
    methods = sorted({m.method for m in method_summaries})
    values = [mean([m.cumulative_improvement for m in method_summaries if m.method == method]) for method in methods]

    plt.figure(figsize=(8, 4))
    plt.bar(methods, values)
    plt.ylabel("Cumulative Improvement")
    plt.title("Cumulative Improvement by Method")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def _plot_open_world_metrics(method_summaries: list[MethodSummary], out: Path) -> Path:
    methods = sorted({m.method for m in method_summaries})
    heldout_vals = [mean([m.heldout_predictive_accuracy for m in method_summaries if m.method == method]) for method in methods]
    ood_vals = [mean([m.ood_predictive_accuracy for m in method_summaries if m.method == method]) for method in methods]
    stress_vals = [mean([m.stress_predictive_accuracy for m in method_summaries if m.method == method]) for method in methods]

    x = list(range(len(methods)))
    width = 0.25

    plt.figure(figsize=(10, 4))
    plt.bar([i - width for i in x], heldout_vals, width=width, label="heldout")
    plt.bar(x, ood_vals, width=width, label="ood")
    plt.bar([i + width for i in x], stress_vals, width=width, label="stress")
    plt.ylabel("Accuracy")
    plt.title("Heldout vs OOD vs Stress Accuracy")
    plt.xticks(x, methods, rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out
