"""Prepare BrierBench task files from ForecastBench question + resolution sets.

Downloads question sets and resolution sets from the ForecastBench datasets
repo, joins them by (id, source), and writes train.json / test.json files
suitable for the BrierBench OpenReward environment.

Usage:
    uv run brierbench/prepare_tasks.py --output-dir brierbench/data
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from loguru import logger

DATASETS_REPO = "forecastingresearch/forecastbench-datasets"
RAW_BASE = f"https://raw.githubusercontent.com/{DATASETS_REPO}/main"


def fetch_raw(path: str) -> str:
    """Fetch a file from the datasets repo."""
    url = f"{RAW_BASE}/{path}"
    result = subprocess.run(
        ["curl", "-sL", url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch {url}: {result.stderr}")
    return result.stdout


def list_available_files(subdir: str) -> list[str]:
    """List JSON files in a datasets repo subdirectory via GitHub API."""
    api_url = (
        f"https://api.github.com/repos/{DATASETS_REPO}"
        f"/contents/datasets/{subdir}"
    )
    result = subprocess.run(
        ["curl", "-sL", api_url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list {subdir}: {result.stderr}")

    entries = json.loads(result.stdout)
    return [
        e["name"]
        for e in entries
        if isinstance(e, dict) and e["name"].endswith(".json")
    ]


def load_question_set(filename: str) -> list[dict]:
    """Load a question set JSON file."""
    content = fetch_raw(f"datasets/question_sets/{filename}")
    data = json.loads(content)
    return data.get("questions", [])


def load_resolution_set(filename: str) -> dict[tuple[str, str], dict]:
    """Load a resolution set and index by (id, source)."""
    content = fetch_raw(f"datasets/resolution_sets/{filename}")
    data = json.loads(content)

    by_key: dict[tuple[str, str], dict] = {}
    for r in data.get("resolutions", []):
        if not r.get("resolved"):
            continue
        if isinstance(r.get("id"), list):
            continue
        key = (str(r["id"]), r["source"])
        by_key[key] = r
    return by_key


def build_tasks(
    questions: list[dict],
    resolutions: dict[tuple[str, str], dict],
    question_set_date: str,
) -> list[dict]:
    """Join questions with resolutions to build task specs."""
    tasks = []
    for q in questions:
        qid = str(q["id"])
        source = q["source"]
        key = (qid, source)

        r = resolutions.get(key)
        if r is None:
            continue

        # Determine dates
        start_date = q.get("freeze_datetime", question_set_date)
        if "T" in start_date:
            start_date = start_date.split("T")[0]

        # For market questions: use market close as resolution date
        # For dataset questions: use first resolution date
        if q.get("market_info_close_datetime"):
            res_date = q["market_info_close_datetime"]
            if "T" in res_date:
                res_date = res_date.split("T")[0]
        elif q.get("resolution_dates"):
            res_dates = q["resolution_dates"]
            if isinstance(res_dates, list) and res_dates:
                res_date = res_dates[0]
                if "T" in res_date:
                    res_date = res_date.split("T")[0]
            else:
                continue
        else:
            res_date_raw = r.get("resolution_date", "")
            if not res_date_raw:
                continue
            res_date = res_date_raw.split("T")[0]

        # Skip if resolution date is before or same as start
        if res_date <= start_date:
            continue

        outcome = float(r["resolved_to"])

        task = {
            "id": f"{question_set_date}_{source}_{qid}",
            "question": q["question"],
            "background": q.get("background", ""),
            "resolution_criteria": q.get("resolution_criteria", ""),
            "source": source,
            "start_date": start_date,
            "resolution_date": res_date,
            "outcome": outcome,
        }
        tasks.append(task)

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare BrierBench task files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="brierbench/data",
        help="Directory to write train.json and test.json",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of tasks to hold out for test split",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover available files
    logger.info("Listing available question and resolution sets...")
    q_files = list_available_files("question_sets")
    r_files = list_available_files("resolution_sets")

    # Filter to dated question sets (YYYY-MM-DD-llm.json)
    q_dated = sorted([
        f for f in q_files
        if f[0].isdigit() and f.endswith("-llm.json")
    ])
    logger.info(f"Found {len(q_dated)} question sets")

    # Build resolution index from all resolution sets
    logger.info("Loading resolution sets...")
    all_resolutions: dict[tuple[str, str], dict] = {}
    for rf in r_files:
        if not rf[0].isdigit():
            continue
        try:
            resolutions = load_resolution_set(rf)
            all_resolutions.update(resolutions)
            logger.info(f"  {rf}: {len(resolutions)} resolved")
        except Exception as e:
            logger.warning(f"  Skipping {rf}: {e}")

    logger.info(
        f"Total unique resolved questions: {len(all_resolutions)}"
    )

    # Build tasks from all question sets
    all_tasks: list[dict] = []
    seen_ids: set[str] = set()

    for qf in q_dated:
        date_str = qf.replace("-llm.json", "")
        try:
            questions = load_question_set(qf)
        except Exception as e:
            logger.warning(f"  Skipping {qf}: {e}")
            continue

        tasks = build_tasks(questions, all_resolutions, date_str)

        # Deduplicate by task id
        new_tasks = [t for t in tasks if t["id"] not in seen_ids]
        seen_ids.update(t["id"] for t in new_tasks)
        all_tasks.extend(new_tasks)
        logger.info(
            f"  {qf}: {len(questions)} questions -> "
            f"{len(new_tasks)} new tasks"
        )

    logger.info(f"Total tasks: {len(all_tasks)}")

    # Sort by start_date so test split contains the most recent questions
    all_tasks.sort(key=lambda t: (t["start_date"], t["id"]))

    # Split: earliest questions are train, most recent are test
    n_test = max(1, int(len(all_tasks) * args.test_fraction))
    train_tasks = all_tasks[: len(all_tasks) - n_test]
    test_tasks = all_tasks[len(all_tasks) - n_test :]

    # Write output
    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"

    train_path.write_text(json.dumps(train_tasks, indent=2))
    test_path.write_text(json.dumps(test_tasks, indent=2))

    logger.info(
        f"Wrote {len(train_tasks)} train tasks to {train_path}"
    )
    logger.info(
        f"Wrote {len(test_tasks)} test tasks to {test_path}"
    )


if __name__ == "__main__":
    main()
