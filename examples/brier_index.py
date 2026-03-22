"""Compute difficulty-adjusted Brier Index using ForecastBench fixed effects.

Downloads question fixed effects from ForecastBench and computes the
Brier Index as: BI = (1 - (mean_brier - mean_question_FE)) * 100

Also loads processed forecast sets to compare against published models.

Usage:
    uv run brierbench/examples/brier_index.py \
        /tmp/brierbench-results/zero_shot_hosted.json \
        /tmp/brierbench-results/forecast_skill_hosted.json

    # With pre-downloaded data:
    uv run brierbench/examples/brier_index.py \
        --fixed-effects /tmp/forecastbench-data/question_fixed_effects.json \
        --forecast-dir /tmp/forecastbench-data/forecastbench-processed-forecast-sets \
        /tmp/brierbench-results/*.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

_DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})_")
_SOURCE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_([^_]+)_")
_ID_SUFFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_[^_]+_(.+)$")

FE_URL = (
    "https://www.forecastbench.org/assets/data/"
    "question-fixed-effects/question_fixed_effects.2026-03-22.json"
)


def _load_fixed_effects(path: Path | None) -> dict[tuple[str, str], float]:
    """Load question fixed effects, keyed by (source, id)."""
    if path is None:
        content = subprocess.run(
            ["curl", "-sL", FE_URL],
            capture_output=True,
            text=True,
        ).stdout
        data = json.loads(content)
    else:
        data = json.loads(path.read_text())

    fe = {}
    for q in data:
        key = (q["source"], str(q["id"]))
        fe[key] = q["question_fixed_effect"]
    return fe


def _load_published_models(
    forecast_dir: Path | None,
) -> dict[str, list[dict]]:
    """Load per-model forecasts from processed forecast sets."""
    if forecast_dir is None:
        return {}

    models: dict[str, list[dict]] = {}
    for date_dir in sorted(forecast_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for f in date_dir.glob("*.json"):
            data = json.loads(f.read_text())
            model_name = data.get("model", f.stem)
            if model_name not in models:
                models[model_name] = []
            for fc in data.get("forecasts", []):
                if fc.get("resolved") and fc.get("forecast") is not None:
                    models[model_name].append(fc)
    return models


def _compute_brier_index(
    briers_and_fes: list[tuple[float, float]],
) -> dict:
    """Compute Brier Index from (brier, fixed_effect) pairs."""
    if not briers_and_fes:
        return {"n": 0}
    briers = np.array([b for b, _ in briers_and_fes])
    fes = np.array([fe for _, fe in briers_and_fes])
    adjusted = briers - fes
    return {
        "n": len(briers),
        "mean_brier": float(np.mean(briers)),
        "mean_fe": float(np.mean(fes)),
        "mean_adjusted": float(np.mean(adjusted)),
        "brier_index": float((1 - np.mean(adjusted)) * 100),
        "raw_bi": float((1 - np.mean(briers)) * 100),
    }


def _score_our_run(
    path: Path,
    fe: dict[tuple[str, str], float],
) -> dict:
    """Score a BrierBench result file against fixed effects."""
    data = json.loads(path.read_text())
    strategy = data["strategy"]
    model = data.get("model", "unknown")

    pairs = []
    n_missing_fe = 0
    for r in data["results"]:
        if r.get("reward") is None:
            continue
        brier = 1 - r["reward"]
        m = _SOURCE_PREFIX.match(r["task_id"])
        m2 = _ID_SUFFIX.match(r["task_id"])
        if not m or not m2:
            continue
        source = m.group(1)
        qid = m2.group(1)
        key = (source, qid)
        if key in fe:
            pairs.append((brier, fe[key]))
        else:
            n_missing_fe += 1

    stats = _compute_brier_index(pairs)
    stats["strategy"] = strategy
    stats["model"] = model
    stats["n_missing_fe"] = n_missing_fe
    stats["n_total"] = sum(
        1 for r in data["results"] if r.get("reward") is not None
    )
    return stats


def _score_published_model(
    model_name: str,
    forecasts: list[dict],
    fe: dict[tuple[str, str], float],
) -> dict:
    """Score a published model's forecasts against fixed effects."""
    pairs = []
    for fc in forecasts:
        if fc.get("forecast") is None or not fc.get("resolved"):
            continue
        brier = (fc["forecast"] - fc["resolved_to"]) ** 2
        key = (fc["source"], str(fc["id"]))
        if key in fe:
            pairs.append((brier, fe[key]))

    stats = _compute_brier_index(pairs)
    stats["model"] = model_name
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Brier Index with ForecastBench fixed effects"
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--fixed-effects", type=Path, default=None)
    parser.add_argument("--forecast-dir", type=Path, default=None)
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Show top N published models for comparison",
    )
    args = parser.parse_args()

    print("Loading question fixed effects...")
    fe = _load_fixed_effects(args.fixed_effects)
    print(f"  {len(fe)} questions with fixed effects\n")

    # Score our runs
    our_results = []
    for path in args.files:
        stats = _score_our_run(path, fe)
        our_results.append(stats)

    # Score published models
    published = {}
    if args.forecast_dir:
        print("Loading published model forecasts...")
        models = _load_published_models(args.forecast_dir)
        print(f"  {len(models)} models loaded\n")
        for model_name, forecasts in models.items():
            stats = _score_published_model(model_name, forecasts, fe)
            if stats["n"] > 0:
                published[model_name] = stats

    # Print our results
    print(f"{'=' * 72}")
    print("  OUR RUNS")
    print(f"{'=' * 72}")
    print(
        f"  {'Strategy':<28s} {'BI':>6s} {'Raw BI':>7s} "
        f"{'Brier':>7s} {'Adj':>7s} {'FE':>6s} {'n':>5s}"
    )
    print(f"  {'-' * 68}")
    for s in sorted(our_results, key=lambda x: -x.get("brier_index", 0)):
        if s["n"] == 0:
            print(f"  {s['strategy']:<28s} (no matched questions)")
            continue
        print(
            f"  {s['strategy']:<28s} "
            f"{s['brier_index']:>5.1f} "
            f"{s['raw_bi']:>6.1f}  "
            f"{s['mean_brier']:>6.4f} "
            f"{s['mean_adjusted']:>6.4f} "
            f"{s['mean_fe']:>5.3f} "
            f"{s['n']:>5d}"
        )
    print()

    # Print published comparison
    if published:
        ranked = sorted(
            published.items(),
            key=lambda x: -x[1].get("brier_index", 0),
        )
        print(f"{'=' * 72}")
        print(f"  FORECASTBENCH LEADERBOARD (top {args.top_n})")
        print(f"{'=' * 72}")
        print(
            f"  {'Model':<40s} {'BI':>6s} "
            f"{'Brier':>7s} {'n':>6s}"
        )
        print(f"  {'-' * 58}")
        for model_name, s in ranked[: args.top_n]:
            print(
                f"  {model_name:<40s} "
                f"{s['brier_index']:>5.1f} "
                f"{s['mean_brier']:>6.4f} "
                f"{s['n']:>6d}"
            )
        print()


if __name__ == "__main__":
    main()
