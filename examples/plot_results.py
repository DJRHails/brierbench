"""Plot BrierBench evaluation results: Brier score over question date.

Reads one or more batch_eval output JSON files and plots mean Brier
score by question set date, with a dashed line at the model's training
cutoff.

Usage:
    uv run brierbench/examples/plot_results.py \
        /tmp/brierbench-results/zero_shot.json \
        /tmp/brierbench-results/zero_shot_research.json \
        --cutoff 2025-01-01 \
        --output figures/brierbench_brier_by_date.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/dh/.claude/skills/graph-design")
from graph_design import colors, finalize, label_lines, set_theme

_DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})_")

MODEL_CUTOFFS = {
    "google/gemini-3-flash-preview": "2025-01-01",
    "google/gemini-2.5-flash": "2025-01-01",
    "deepseek/deepseek-v3.2": "2025-02-01",
    "x-ai/grok-4": "2025-04-01",
}


def _extract_date(task_id: str) -> str | None:
    m = _DATE_PREFIX.match(task_id)
    return m.group(1) if m else None


def _load_run(path: Path) -> dict:
    data = json.loads(path.read_text())
    strategy = data["strategy"]
    model = data.get("model", "unknown")

    by_date: dict[str, list[float]] = defaultdict(list)
    all_dates: list[date] = []
    all_briers: list[float] = []
    for r in data["results"]:
        if r.get("reward") is None:
            continue
        d = _extract_date(r["task_id"])
        if d is None:
            continue
        brier = 1.0 - r["reward"]
        by_date[d].append(brier)
        all_dates.append(date.fromisoformat(d))
        all_briers.append(brier)

    agg_dates = sorted(by_date.keys())
    means = [np.mean(by_date[d]) for d in agg_dates]
    ci_95 = [
        1.96 * np.std(by_date[d]) / max(1, np.sqrt(len(by_date[d])))
        for d in agg_dates
    ]
    return {
        "strategy": strategy,
        "model": model,
        "agg_dates": [date.fromisoformat(d) for d in agg_dates],
        "mean_brier": np.array(means),
        "ci_95": np.array(ci_95),
        "scatter_dates": all_dates,
        "scatter_briers": all_briers,
        "overall_mean": np.mean(all_briers) if all_briers else float("nan"),
        "n_scored": len(all_briers),
    }


def _plot_panel(ax, runs: list[dict], palette: list, cutoff: str | None) -> None:
    """Plot runs onto a single axes panel."""
    for i, run in enumerate(runs):
        c = palette[i % len(palette)]
        label = (
            f"{run['strategy']} "
            f"(brier={run['overall_mean']:.3f}, n={run['n_scored']})"
        )

        ax.scatter(
            run["scatter_dates"],
            run["scatter_briers"],
            color=c,
            alpha=0.25,
            s=12,
            zorder=2,
            edgecolors="none",
        )
        ax.fill_between(
            run["agg_dates"],
            run["mean_brier"] - run["ci_95"],
            run["mean_brier"] + run["ci_95"],
            color=c,
            alpha=0.15,
            zorder=3,
        )
        ax.plot(
            run["agg_dates"],
            run["mean_brier"],
            color=c,
            linewidth=2,
            label=label,
            zorder=4,
        )

    if cutoff:
        cd = date.fromisoformat(cutoff)
        ax.axvline(cd, color="#666666", linestyle="--", linewidth=1.2, zorder=0)
        ax.text(
            cd,
            ax.get_ylim()[1] * 0.95,
            " Model cutoff ",
            fontsize=8,
            color="#666666",
            ha="right",
            va="top",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot BrierBench results by question date"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Batch eval JSON output files",
    )
    parser.add_argument(
        "--cutoff",
        default=None,
        help="Model training cutoff date (YYYY-MM-DD). "
        "Auto-detected from model name if omitted.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save plot to file instead of showing",
    )
    args = parser.parse_args()

    runs = [_load_run(p) for p in args.files]

    # Group by model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_model[run["model"]].append(run)

    models = sorted(by_model.keys())
    n_models = len(models)
    palette = list(colors)

    set_theme()

    if n_models == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]
        fig.subplots_adjust(top=0.68, bottom=0.14, left=0.08, right=0.62)
    else:
        fig, axes = plt.subplots(
            n_models, 1,
            figsize=(7, 4 * n_models),
            sharex=True,
        )
        fig.subplots_adjust(
            top=1 - 0.12 / n_models,
            bottom=0.08,
            left=0.08,
            right=0.62,
            hspace=0.25,
        )

    for ax, model in zip(axes, models):
        model_runs = by_model[model]
        cutoff = args.cutoff or MODEL_CUTOFFS.get(model)
        _plot_panel(ax, model_runs, palette, cutoff)
        label_lines(ax, x_offset=8)

        short_model = model.split("/")[-1] if "/" in model else model
        cutoff_str = f" (cutoff {cutoff})" if cutoff else ""
        ax.set_title(
            f"{short_model}{cutoff_str}",
            fontsize=10, loc="left", pad=8,
        )
        ax.set_ylabel("Brier score")

    finalize(
        axes[0],
        title="LLM forecasting accuracy by question date",
        descriptor="BrierBench, Brier score (lower is better), "
        "by question set date",
        source="Source: BrierBench evaluation via OpenReward",
        y_start=0.075 if n_models > 1 else 0.0,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
