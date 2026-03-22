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
    """Run the OpenRouter agent loop until finished or no tool calls.

    Returns dict with 'reward', 'finished', 'messages', 'n_tool_calls'.
    """
    tools = environment.list_tools(format="openrouter")

    if user_message is None:
        prompt = session.get_prompt()
        user_message = prompt[0].text

    input_list: list[dict] = []
    if system_message:
        input_list.append({"role": "system", "content": system_message})
    input_list.append({"role": "user", "content": user_message})

    finished = False
    reward = None
    n_tool_calls = 0

    while not finished:
        response = oai_client.chat.completions.create(
            model=MODEL_NAME,
            tools=tools,
            messages=input_list,
        )

        assistant_message = {
            "role": "assistant",
            "content": response.choices[0].message.content,
            "tool_calls": response.choices[0].message.tool_calls,
        }
        input_list.append(assistant_message)

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            break

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            n_tool_calls += 1

            print(f"[{tool_name}] {json.dumps(tool_args)[:120]}")

            try:
                tool_result = session.call_tool(tool_name, tool_args)
            except Exception as e:
                error_text = f"Tool error: {e}"
                print(f"  -> {error_text[:200]}")
                input_list.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"error": error_text}),
                })
                continue

            result_text = tool_result.blocks[0].text
            print(f"  -> {result_text[:200]}")

            if tool_result.reward is not None:
                reward = tool_result.reward
                print(f"  reward={reward:.4f} finished={tool_result.finished}")

            finished = tool_result.finished

            input_list.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"result": result_text}),
            })

            if finished:
                break

    # Print final text if the model gave one
    last = input_list[-1]
    if last.get("role") == "assistant" and last.get("content"):
        print(f"\n[assistant] {last['content'][:500]}")

    return {
        "reward": reward,
        "finished": finished,
        "messages": input_list,
        "n_tool_calls": n_tool_calls,
    }
