"""Batch evaluation runner for BrierBench examples.

Runs N tasks at given concurrency for a specified example strategy.

Usage:
    uv run brierbench/examples/batch_eval.py \
        --strategy zero_shot \
        --n-samples 100 \
        --concurrency 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from _common import (
    MODEL_NAME,
    environment,
    get_secrets,
    run_agent_loop,
    upload_forecast_skill,
)

from forecast_skill import build_system_message as forecast_skill_system
from zero_shot import build_zero_shot_prompt
from zero_shot_research import build_research_prompt


def _run_task(strategy: str, task, run_name: str) -> dict:
    """Run a single task with the given strategy. Returns result dict."""
    spec = task.task_spec
    secrets = get_secrets()

    with environment.session(task=task, secrets=secrets) as session:
        if strategy == "forecast_skill":
            upload_forecast_skill(session)
            result = run_agent_loop(
                session,
                run_name=run_name,
                rollout_name=spec["id"],
                system_message=forecast_skill_system(),
            )
        elif strategy == "zero_shot_research":
            result = run_agent_loop(
                session,
                run_name=run_name,
                rollout_name=spec["id"],
                user_message=build_research_prompt(spec),
            )
        else:
            result = run_agent_loop(
                session,
                run_name=run_name,
                rollout_name=spec["id"],
                user_message=build_zero_shot_prompt(spec),
            )

    return {
        "task_id": spec["id"],
        "reward": result["reward"],
        "finished": result["finished"],
        "n_tool_calls": result["n_tool_calls"],
        "n_messages": len(result["messages"]),
    }


def _run_batch(
    strategy: str,
    tasks: list,
    run_name: str,
    concurrency: int,
) -> list[dict]:
    """Run tasks with process-based concurrency."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_run_task, strategy, task, run_name): task
            for task in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            spec = task.task_spec
            try:
                result = future.result()
                results.append(result)
                reward = result["reward"]
                brier = 1 - reward if reward is not None else None
                elapsed = time.time() - t0
                print(
                    f"[{i}/{len(tasks)}] {spec['id'][:60]} "
                    f"reward={reward} brier={brier} "
                    f"({elapsed:.0f}s elapsed)"
                )
            except Exception as e:
                print(f"[{i}/{len(tasks)}] {spec['id'][:60]} FAILED: {e}")
                results.append({
                    "task_id": spec["id"],
                    "reward": None,
                    "finished": False,
                    "error": str(e),
                })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch BrierBench evaluation"
    )
    parser.add_argument(
        "--strategy",
        choices=["zero_shot", "zero_shot_research", "forecast_skill"],
        required=True,
    )
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write results JSON",
    )
    args = parser.parse_args()

    all_tasks = environment.list_tasks(split=args.split)
    rng = random.Random(args.seed)
    tasks = rng.sample(all_tasks, min(args.n_samples, len(all_tasks)))

    run_name = f"brierbench-{args.strategy}-{args.n_samples}"
    print(f"Strategy: {args.strategy}")
    print(f"Model: {MODEL_NAME}")
    print(f"Tasks: {len(tasks)} (from {len(all_tasks)} in {args.split})")
    print(f"Concurrency: {args.concurrency}")
    print(f"Run: {run_name}")
    print()

    t0 = time.time()
    results = _run_batch(
        args.strategy, tasks, run_name, args.concurrency
    )
    elapsed = time.time() - t0

    # Summary
    rewards = [r["reward"] for r in results if r.get("reward") is not None]
    briers = [1 - r for r in rewards]
    n_failed = sum(1 for r in results if r.get("reward") is None)

    print(f"\n{'=' * 60}")
    print(f"  {args.strategy} | {len(tasks)} tasks | {elapsed:.0f}s")
    print(f"{'=' * 60}")
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        mean_brier = sum(briers) / len(briers)
        print(f"  Mean reward:      {mean_reward:.4f}")
        print(f"  Mean Brier:       {mean_brier:.4f}")
        print(f"  Scored:           {len(rewards)}")
    print(f"  Failed:           {n_failed}")
    print()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "strategy": args.strategy,
            "model": MODEL_NAME,
            "split": args.split,
            "n_samples": len(tasks),
            "concurrency": args.concurrency,
            "elapsed_seconds": elapsed,
            "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            "mean_brier": sum(briers) / len(briers) if briers else None,
            "n_scored": len(rewards),
            "n_failed": n_failed,
            "results": results,
        }, indent=2))
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
