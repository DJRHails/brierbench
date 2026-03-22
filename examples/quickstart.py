"""BrierBench quickstart: run a single task with Gemini via OpenRouter.

Uploads forecast-skill scripts into the sandbox at session start,
then runs an agent loop that can use them for distribution generation.

Usage:
    export OPENREWARD_API_KEY='...'
    export OPENROUTER_API_KEY='...'
    export EXA_API_KEY='...'
    uv run brierbench/examples/quickstart.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openreward import OpenReward

load_dotenv()

FORECAST_SKILL_DIR = Path("forecast-skill")
MODEL_NAME = "google/gemini-3-flash-preview"

or_client = OpenReward()
oai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

environment = or_client.environments.get(
    name="brierbench",
    base_url="http://localhost:8081",
)
tasks = environment.get_task_range(split="train", start=0, stop=1)
tools = environment.list_tools(format="openrouter")

secrets = {
    "EXA_API_KEY": (os.environ["EXA_API_KEY"], ["api.exa.ai"]),
}


def upload_forecast_skill(session) -> None:
    """Upload forecast-skill scripts and docs into the sandbox via bash."""
    session.call_tool("bash", {"command": "mkdir -p /forecast-skill"})

    # Upload SKILL.md (docs the agent can read via bash)
    skill_md = (FORECAST_SKILL_DIR / "SKILL.md").read_text()
    session.call_tool("bash", {
        "command": (
            "cat <<'FORECAST_SKILL_EOF' > /forecast-skill/SKILL.md\n"
            f"{skill_md}\n"
            "FORECAST_SKILL_EOF"
        ),
    })

    # Upload scripts
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

    n = sum(1 for s in scripts_dir.glob("*.py") if not s.name.startswith("__"))
    print(f"Uploaded SKILL.md + {n} scripts to /forecast-skill/")


def build_system_message() -> str:
    """System message pointing the agent at the forecast-skill docs."""
    return (
        "Forecast distribution scripts are available at "
        "/forecast-skill/. Read /forecast-skill/SKILL.md "
        "with `bash` for usage docs and examples."
    )


example_task = tasks[0]
with environment.session(task=example_task, secrets=secrets) as session:
    # Setup: upload scripts (invisible to the model)
    upload_forecast_skill(session)

    # Build conversation — point the agent at the docs file
    prompt = session.get_prompt()
    input_list = [
        {"role": "system", "content": build_system_message()},
        {"role": "user", "content": prompt[0].text},
    ]
    finished = False

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

            print(f"[{tool_name}] {json.dumps(tool_args)[:120]}")

            tool_result = session.call_tool(tool_name, tool_args)

            reward = tool_result.reward
            finished = tool_result.finished

            result_text = tool_result.blocks[0].text
            print(f"  -> {result_text[:200]}")
            if reward is not None:
                print(f"  reward={reward:.4f} finished={finished}")

            tool_result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"result": result_text}),
            }
            input_list.append(tool_result_message)

            if finished:
                break

    # Print final assistant message if it has text content
    last = input_list[-1]
    if last.get("role") == "assistant" and last.get("content"):
        print(f"\n[assistant] {last['content'][:500]}")

    print(f"\nTotal messages: {len(input_list)}")
