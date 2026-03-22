"""Shared agent loop and utilities for BrierBench examples."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
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
    system_message: str | None = None,
    user_message: str | None = None,
) -> dict:
    """Run the agent loop using OpenAI Responses API format.

    When the model stops calling tools, auto-advances to resolution
    via advance_time so a reward is always produced.

    Returns dict with 'reward', 'finished', 'messages', 'n_tool_calls'.
    """
    tools = environment.list_tools(format="openai")

    if user_message is None:
        prompt = session.get_prompt()
        user_message = prompt[0].text

    input_list: list = []
    if system_message:
        input_list.append({
            "role": "developer",
            "content": system_message,
        })
    input_list.append({"role": "user", "content": user_message})

    finished = False
    reward = None
    n_tool_calls = 0

    while not finished:
        response = oai_client.responses.create(
            model=MODEL_NAME,
            tools=tools,
            input=input_list,
        )

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
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({"error": error_text}),
                    })
                    continue

                result_text = tool_result.blocks[0].text
                print(f"  -> {result_text[:200]}")

                reward = tool_result.reward
                finished = tool_result.finished

                if reward is not None:
                    print(f"  reward={reward:.4f} finished={finished}")

                input_list.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps({"result": result_text}),
                })

                if finished:
                    break

        # No function calls in this response — model is done
        if not any(
            getattr(item, "type", None) == "function_call"
            for item in response.output
        ):
            break

    # If model stopped without finishing, force advance to resolution
    if not finished:
        print("\n[auto] Model stopped — advancing to resolution...")
        result = session.call_tool("advance_time", {"days": 365})
        text = result.blocks[0].text
        print(f"  -> {text[:200]}")
        reward = result.reward
        finished = result.finished
        if reward is not None:
            print(f"  reward={reward:.4f} finished={finished}")

    return {
        "reward": reward,
        "finished": finished,
        "messages": input_list,
        "n_tool_calls": n_tool_calls,
    }
