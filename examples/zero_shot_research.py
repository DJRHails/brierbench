"""BrierBench zero-shot with research: search and update over time.

Like zero_shot.py but encourages the agent to use exa_search for
evidence gathering, advance_time to simulate passage of time, and
update predictions as new information becomes available.

Usage:
    uv run brierbench/examples/zero_shot_research.py
"""

from _common import (
    environment,
    get_secrets,
    run_agent_loop,
)

ZERO_SHOT_RESEARCH = """\
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

## Instructions

1. RESEARCH first. Use `exa_search` to find current evidence relevant \
to this question. Look for official sources, news reports, prediction \
market prices, and expert analysis.

2. SUBMIT an initial prediction using `submit_prediction` based on your \
research.

3. ADVANCE TIME using `advance_time` to move forward (e.g. 14-30 days \
at a time).

4. RESEARCH AGAIN after advancing — new information may be available. \
Search for updates, developments, or changed conditions.

5. UPDATE your prediction with `submit_prediction` if the evidence \
warrants a change. Later predictions carry more weight in scoring.

6. REPEAT steps 3-5 until you reach the resolution date or are \
confident your prediction is stable.

Your reward is based on a time-weighted Brier score — predictions made \
later count more. Iterating and updating as you gather new information \
will improve your score.\
"""


def build_research_prompt(task_spec: dict) -> str:
    """Format the research-updating prompt from the task spec."""
    return ZERO_SHOT_RESEARCH.format(
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
        user_message=build_research_prompt(spec),
    )

print(f"\nReward: {result['reward']}")
print(f"Tool calls: {result['n_tool_calls']}")
print(f"Messages: {len(result['messages'])}")
