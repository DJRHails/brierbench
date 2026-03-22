"""BrierBench zero-shot baseline: ForecastBench-style prompting.

Uses the ZERO_SHOT_MARKET / ZERO_SHOT_DATASET prompt templates from
Halawi et al. (2024) — no tools, no search, just a single prediction
based on the model's parametric knowledge.

Usage:
    uv run brierbench/examples/zero_shot.py
"""

from _common import (
    environment,
    get_secrets,
    run_agent_loop,
)

ZERO_SHOT_MARKET = """\
You are an expert superforecaster, familiar with the work of Tetlock and \
others. Make a prediction of the probability that the question will be \
resolved as true. You MUST give a probability estimate between 0 and 1 \
UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the \
base rate, but return a number between 0 and 1.

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Today's Date: {today_date}

Question resolution date: {resolution_date}

You have access to tools. Use `submit_prediction` to submit your \
probability estimate. You MUST call submit_prediction with your answer.
"""

ZERO_SHOT_DATASET = """\
You are an expert superforecaster, familiar with the work of Tetlock and \
others. Make a prediction of the probability that the question will be \
resolved as true. You MUST give a probability estimate between 0 and 1 \
UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the \
base rate, but return a number between 0 and 1.

You're going to predict the probability of the following potential outcome.

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Today's Date: {today_date}

Question resolution date: {resolution_date}

You have access to tools. Use `submit_prediction` to submit your \
probability estimate. You MUST call submit_prediction with your answer.
"""


def build_zero_shot_prompt(task_spec: dict) -> str:
    """Format the zero-shot prompt from the task spec."""
    source = task_spec.get("source", "")
    is_market = source in {"manifold", "metaculus", "infer", "polymarket"}

    template = ZERO_SHOT_MARKET if is_market else ZERO_SHOT_DATASET
    return template.format(
        question=task_spec["question"],
        background=task_spec.get("background", ""),
        resolution_criteria=task_spec.get("resolution_criteria", ""),
        today_date=task_spec["start_date"],
        resolution_date=task_spec["resolution_date"],
    )


tasks = environment.get_task_range(split="train", start=0, stop=1)
task = tasks[0]
spec = task.task_spec

print(f"Question: {spec['question'][:120]}...")
print(f"Source: {spec['source']}")
print(f"Dates: {spec['start_date']} -> {spec['resolution_date']}")
print(f"Outcome: {'YES' if spec['outcome'] == 1.0 else 'NO'}\n")

with environment.session(task=task, secrets=get_secrets()) as session:
    result = run_agent_loop(
        session,
        run_name="brierbench-zero-shot",
        rollout_name=spec["id"],
        user_message=build_zero_shot_prompt(spec),
    )

print(f"\nReward: {result['reward']}")
print(f"Tool calls: {result['n_tool_calls']}")
print(f"Messages: {len(result['messages'])}")
