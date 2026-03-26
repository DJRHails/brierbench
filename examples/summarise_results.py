"""Summarise BrierBench evaluation results with Brier Index.

Brier Index = (1 - brier_score) * 100, on a 0–100 scale (higher = better).
Breaks down by pre/post training cutoff and by source type.

Usage:
    uv run brierbench/examples/summarise_results.py \
        /tmp/brierbench-results/zero_shot_hosted.json \
        /tmp/brierbench-results/zero_shot_research_hosted.json \
        /tmp/brierbench-results/zero_shot_hosted.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

_DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})_")
_SOURCE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_([^_]+)_")

MARKET_SOURCES = {"manifold", "metaculus", "infer", "polymarket"}

MODEL_CUTOFFS = {
    "google/gemini-3-flash-preview": "2025-01-01",
    "google/gemini-2.5-flash": "2025-01-01",
    "deepseek/deepseek-v3.2": "2025-02-01",
    "x-ai/grok-4": "2025-04-01",
}


def _extract_date(task_id: str) -> date | None:
    m = _DATE_PREFIX.match(task_id)
    return date.fromisoformat(m.group(1)) if m else None


def _extract_source(task_id: str) -> str | None:
    m = _SOURCE_PREFIX.match(task_id)
    return m.group(1) if m else None


def _brier_index(brier: float) -> float:
    return (1 - brier) * 100


def _summarise_group(briers: list[float], label: str) -> dict:
    if not briers:
        return {"label": label, "n": 0}
    arr = np.array(briers)
    return {
        "label": label,
        "n": len(arr),
        "mean_brier": float(np.mean(arr)),
        "brier_index": float(_brier_index(np.mean(arr))),
        "std": float(np.std(arr)),
        "median_brier": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def _print_group(g: dict) -> None:
    if g["n"] == 0:
        print(f"  {g['label']:.<30s} (no data)")
        return
    print(
        f"  {g['label']:.<30s} "
        f"Brier Index {g['brier_index']:5.1f}  "
        f"(brier {g['mean_brier']:.4f} "
        f"± {g['std']:.4f}, "
        f"median {g['median_brier']:.4f}, "
        f"n={g['n']})"
    )


def summarise_run(path: Path, cutoff_override: str | None = None) -> None:
    data = json.loads(path.read_text())
    strategy = data["strategy"]
    model = data.get("model", "unknown")
    cutoff_str = cutoff_override or MODEL_CUTOFFS.get(model)
    cutoff = date.fromisoformat(cutoff_str) if cutoff_str else None

    scored = [r for r in data["results"] if r.get("reward") is not None]
    failed = [r for r in data["results"] if r.get("reward") is None]

    all_briers = [1 - r["reward"] for r in scored]
    pre_cutoff = []
    post_cutoff = []
    by_source: dict[str, list[float]] = defaultdict(list)
    market_briers = []
    data_briers = []

    for r in scored:
        brier = 1 - r["reward"]
        d = _extract_date(r["task_id"])
        source = _extract_source(r["task_id"])

        if cutoff and d:
            if d < cutoff:
                pre_cutoff.append(brier)
            else:
                post_cutoff.append(brier)

        if source:
            by_source[source].append(brier)
            if source in MARKET_SOURCES:
                market_briers.append(brier)
            else:
                data_briers.append(brier)

    short_model = model.split("/")[-1] if "/" in model else model
    print(f"\n{'=' * 65}")
    print(f"  {strategy} | {short_model} | {path.name}")
    print(f"{'=' * 65}")

    _print_group(_summarise_group(all_briers, "Overall"))
    print()

    if cutoff:
        _print_group(_summarise_group(pre_cutoff, f"Pre-cutoff (<{cutoff_str})"))
        _print_group(_summarise_group(post_cutoff, f"Post-cutoff (≥{cutoff_str})"))
        print()

    _print_group(_summarise_group(market_briers, "Market questions"))
    _print_group(_summarise_group(data_briers, "Data questions"))
    print()

    for source in sorted(by_source.keys()):
        _print_group(_summarise_group(by_source[source], source))
    print()

    print(f"  Scored: {len(scored)}, Failed: {len(failed)}, "
          f"Total: {len(data['results'])}")
    print(f"  Elapsed: {data.get('elapsed_seconds', 0):.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise BrierBench results with Brier Index"
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--cutoff", default=None)
    args = parser.parse_args()

    for path in args.files:
        summarise_run(path, cutoff_override=args.cutoff)

    # Comparison table if multiple files
    if len(args.files) > 1:
        print(f"\n{'=' * 65}")
        print("  COMPARISON")
        print(f"{'=' * 65}")
        print(
            f"  {'Strategy':<25s} {'Brier Index':>12s} "
            f"{'Brier':>8s} {'n':>5s} {'Failed':>7s}"
        )
        print(f"  {'-' * 60}")
        for path in args.files:
            data = json.loads(path.read_text())
            scored = [r for r in data["results"] if r.get("reward") is not None]
            failed = len(data["results"]) - len(scored)
            briers = [1 - r["reward"] for r in scored]
            if briers:
                mean_b = np.mean(briers)
                bi = _brier_index(mean_b)
            else:
                mean_b = float("nan")
                bi = float("nan")
            print(
                f"  {data['strategy']:<25s} {bi:>11.1f}  "
                f"{mean_b:>7.4f} {len(scored):>5d} {failed:>7d}"
            )
        print()


if __name__ == "__main__":
    main()
