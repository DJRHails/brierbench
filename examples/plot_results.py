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
from graph_design import ci_fill, colors, finalize, label_lines, set_theme

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
    for r in data["results"]:
        if r.get("reward") is None:
            continue
        d = _extract_date(r["task_id"])
        if d is None:
            continue
        brier = 1.0 - r["reward"]
        by_date[d].append(brier)

    dates = sorted(by_date.keys())
    means = [np.mean(by_date[d]) for d in dates]
    stderrs = [
        np.std(by_date[d]) / max(1, np.sqrt(len(by_date[d])))
        for d in dates
    ]
    return {
        "strategy": strategy,
        "model": model,
        "dates": [date.fromisoformat(d) for d in dates],
        "mean_brier": np.array(means),
        "stderr": np.array(stderrs),
        "n_per_date": [len(by_date[d]) for d in dates],
        "overall_mean": np.mean([
            b for bs in by_date.values() for b in bs
        ]),
        "n_scored": sum(len(bs) for bs in by_date.values()),
    }


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

    set_theme()
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.68, bottom=0.14, left=0.06, right=0.78)

    cutoff_date = args.cutoff
    palette = list(colors)

    for i, path in enumerate(args.files):
        run = _load_run(path)
        c = palette[i % len(palette)]
        label = (
            f"{run['strategy']} "
            f"(mean={run['overall_mean']:.3f}, n={run['n_scored']})"
        )
        ci_fill(
            ax,
            run["dates"],
            run["mean_brier"] - run["stderr"],
            run["mean_brier"] + run["stderr"],
            color=c,
        )
        ax.plot(
            run["dates"],
            run["mean_brier"],
            color=c,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=label,
        )

        if cutoff_date is None:
            cutoff_date = MODEL_CUTOFFS.get(run["model"])

    if cutoff_date:
        cd = date.fromisoformat(cutoff_date)
        ax.axvline(
            cd,
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            zorder=0,
        )
        ax.text(
            cd,
            ax.get_ylim()[1] * 0.95,
            f" Model cutoff ",
            fontsize=8,
            color="#666666",
            ha="right",
            va="top",
        )

    label_lines(ax, x_offset=8)

    finalize(
        ax,
        title="LLM forecasting accuracy by question date",
        descriptor="BrierBench, Brier score (lower is better), "
        "by question set date",
        source="Source: BrierBench evaluation via OpenReward",
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
