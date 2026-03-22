"""Shared agent loop and utilities for BrierBench examples."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from humane import human_id
from openai import OpenAI
from openreward import OpenReward

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FORECAST_SKILL_DIR = _REPO_ROOT / "forecast-skill"
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemini-3-flash-preview")

or_client = OpenReward()
oai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

environment = or_client.environments.get(
    name="brierbench",
    base_url=os.environ.get("BRIERBENCH_URL", "http://localhost:8081"),
)


def get_secrets() -> dict:
    """Build secrets dict from environment variables."""
    secrets = {}
    exa_key = os.environ.get("EXA_API_KEY")
    if exa_key:
        secrets["EXA_API_KEY"] = (exa_key, ["api.exa.ai"])
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        secrets["OPENROUTER_API_KEY"] = (
            openrouter_key,
            ["openrouter.ai"],
        )
    return secrets


def upload_forecast_skill(session) -> None:
    """Upload forecast-skill scripts and docs into the sandbox."""
    session.call_tool("bash", {"command": "mkdir -p /forecast-skill"})

    skill_md = (FORECAST_SKILL_DIR / "SKILL.md").read_text()
    session.call_tool("bash", {
        "command": (
            "cat <<'FORECAST_SKILL_EOF' > /forecast-skill/SKILL.md\n"
            f"{skill_md}\n"
            "FORECAST_SKILL_EOF"
        ),
    })

    scripts_dir = FORECAST_SKILL_DIR / "scripts"
    for script in sorted(scripts_dir.glob("*.py")):
        if script.name.startswith("__"):
            continue
        content = script.read_text()
        session.call_tool("bash", {
            "command": (
                f"cat <<'FORECAST_SKILL_EOF' > "
                f"/forecast-skill/{script.name}\n"
                f"{content}\n"
                f"FORECAST_SKILL_EOF"
            ),
        })

    n = sum(
        1 for s in scripts_dir.glob("*.py")
        if not s.name.startswith("__")
    )
    print(f"Uploaded SKILL.md + {n} scripts to /forecast-skill/")


def run_agent_loop(
    session,
    *,
    run_name: str = "brierbench-smoke",
    rollout_name: str | None = None,
    split: str = "train",
    system_message: str | None = None,
    user_message: str | None = None,
) -> dict:
    """Run the agent loop using OpenAI Responses API with rollout logging.

    Returns dict with 'reward', 'finished', 'messages', 'n_tool_calls'.
    """
    tools = environment.list_tools(format="openai")

    if user_message is None:
        prompt = session.get_prompt()
        user_message = prompt[0].text

    # Create rollout for logging
    if rollout_name is None:
        rollout_name = f"{run_name}:{human_id()}"
    rollout = or_client.rollout.create(
        run_name=run_name,
        rollout_name=rollout_name,
        environment="DJRHails/brierbench",
        split=split,
        print_messages=True,
    )

    input_list: list = []
    if system_message:
        input_list.append({
            "role": "developer",
            "content": system_message,
        })
    input_list.append({"role": "user", "content": user_message})
    rollout.log_openai_response(input_list[-1])

    finished = False
    reward = None
    n_tool_calls = 0

    while not finished:
        response = oai_client.responses.create(
            model=MODEL_NAME,
            tools=tools,
            input=input_list,
        )
        rollout.log_openai_response(response)

        input_list += response.output

        for item in response.output:
            if item.type == "function_call":
                n_tool_calls += 1
                print(f"[{item.name}] {str(item.arguments)[:120]}")

                try:
                    tool_result = session.call_tool(
                        item.name,
                        json.loads(str(item.arguments)),
                    )
                except Exception as e:
                    error_text = f"Tool error: {e}"
                    print(f"  -> {error_text[:200]}")
                    tool_result_item = {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({"error": error_text}),
                    }
                    input_list.append(tool_result_item)
                    rollout.log_openai_response(tool_result_item)
                    continue

                result_text = tool_result.blocks[0].text
                print(f"  -> {result_text[:200]}")

                reward = tool_result.reward
                finished = tool_result.finished

                if reward is not None:
                    print(f"  reward={reward:.4f} finished={finished}")

                tool_result_item = {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps({"result": result_text}),
                }
                input_list.append(tool_result_item)
                rollout.log_openai_response(
                    tool_result_item,
                    reward=reward,
                    is_finished=finished,
                )

                if finished:
                    break

        # No function calls — re-prompt the model to keep going
        if not any(
            getattr(item, "type", None) == "function_call"
            for item in response.output
        ):
            nudge = {
                "role": "user",
                "content": (
                    "You must call `advance_time` to advance to the "
                    "resolution date. Submit a prediction first with "
                    "`submit_prediction` if you haven't already, then "
                    "call `advance_time` to finish the episode."
                ),
            }
            input_list.append(nudge)
            rollout.log_openai_response(nudge)

    or_client.rollout.close()

    return {
        "reward": reward,
        "finished": finished,
        "messages": input_list,
        "n_tool_calls": n_tool_calls,
    }
