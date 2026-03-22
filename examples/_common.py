"""Shared agent loop and utilities for BrierBench examples."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from humane import human_id
from openai import AsyncOpenAI, OpenAI
from openreward import AsyncOpenReward, OpenReward

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FORECAST_SKILL_DIR = _REPO_ROOT / "forecast-skill"
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemini-3-flash-preview")
_MAX_REPEATS = 3
_BRIERBENCH_URL = os.environ.get("BRIERBENCH_URL") or None
_BRIERBENCH_ENV = os.environ.get("BRIERBENCH_ENV", "DJRHails/brierbench")

# Sync clients (for single-task examples)
or_client = OpenReward()
oai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
environment = or_client.environments.get(
    name=_BRIERBENCH_ENV,
    base_url=_BRIERBENCH_URL,
)

# Async clients (for batch evaluation)
async_or_client = AsyncOpenReward()
async_oai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
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


def _make_tool_output(
    call_id: str,
    *,
    result: str | None = None,
    error: str | None = None,
) -> dict:
    payload = {"error": error} if error else {"result": result}
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(payload),
    }


def upload_forecast_skill(session) -> None:
    """Upload forecast-skill scripts and docs into the sandbox."""
    scripts_dir = FORECAST_SKILL_DIR / "scripts"
    files = {
        "SKILL.md": (FORECAST_SKILL_DIR / "SKILL.md").read_text(),
    }
    for script in sorted(scripts_dir.glob("*.py")):
        if script.name.startswith("__"):
            continue
        files[script.name] = script.read_text()

    parts = ["mkdir -p /forecast-skill/scripts"]
    for name, content in files.items():
        dest = f"/forecast-skill/scripts/{name}"
        if name == "SKILL.md":
            dest = f"/forecast-skill/{name}"
        parts.append(
            f"cat <<'FORECAST_SKILL_EOF' > {dest}\n"
            f"{content}\n"
            f"FORECAST_SKILL_EOF"
        )
    session.call_tool("bash", {"command": " && ".join(parts)})
    print(f"Uploaded {len(files)} files to /forecast-skill/")


async def async_upload_forecast_skill(session) -> None:
    """Async version of upload_forecast_skill."""
    scripts_dir = FORECAST_SKILL_DIR / "scripts"
    files = {
        "SKILL.md": (FORECAST_SKILL_DIR / "SKILL.md").read_text(),
    }
    for script in sorted(scripts_dir.glob("*.py")):
        if script.name.startswith("__"):
            continue
        files[script.name] = script.read_text()

    parts = ["mkdir -p /forecast-skill/scripts"]
    for name, content in files.items():
        dest = f"/forecast-skill/scripts/{name}"
        if name == "SKILL.md":
            dest = f"/forecast-skill/{name}"
        parts.append(
            f"cat <<'FORECAST_SKILL_EOF' > {dest}\n"
            f"{content}\n"
            f"FORECAST_SKILL_EOF"
        )
    await session.call_tool("bash", {"command": " && ".join(parts)})


_NUDGE = (
    "You must call `advance_time` to advance to the "
    "resolution date. Submit a prediction first with "
    "`submit_prediction` if you haven't already, then "
    "call `advance_time` to finish the episode."
)


def run_agent_loop(
    session,
    *,
    run_name: str = "brierbench-smoke",
    rollout_name: str | None = None,
    split: str = "train",
    system_message: str | None = None,
    user_message: str | None = None,
) -> dict:
    """Sync agent loop for single-task examples."""
    tools = environment.list_tools(format="openai")

    if user_message is None:
        user_message = session.get_prompt()[0].text

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
        sys_msg = {"role": "developer", "content": system_message}
        input_list.append(sys_msg)
        rollout.log_openai_response(sys_msg)
    input_list.append({"role": "user", "content": user_message})
    rollout.log_openai_response(input_list[-1])

    finished = False
    reward = None
    n_tool_calls = 0
    last_call_key: str | None = None
    repeat_count = 0

    def _record(tool_out: dict, tool_result) -> None:
        nonlocal reward, finished
        input_list.append(tool_out)
        rollout.log_openai_response(
            tool_out,
            reward=tool_result.reward,
            is_finished=tool_result.finished,
        )
        reward = tool_result.reward
        finished = tool_result.finished
        print(f"  -> {tool_result.blocks[0].text[:200]}")
        if reward is not None:
            print(f"  reward={reward:.4f} finished={finished}")

    while not finished:
        response = oai_client.responses.create(
            model=MODEL_NAME,
            tools=tools,
            input=input_list,
        )
        rollout.log_openai_response(response)
        input_list += response.output

        had_function_call = False
        for item in response.output:
            if item.type != "function_call":
                continue

            had_function_call = True
            n_tool_calls += 1
            call_key = f"{item.name}:{item.arguments}"
            print(f"[{item.name}] {str(item.arguments)[:120]}")

            if call_key == last_call_key:
                repeat_count += 1
            else:
                repeat_count = 0
            last_call_key = call_key

            if repeat_count >= _MAX_REPEATS:
                print(f"  -> Repeated {repeat_count}x, forcing advance")
                result = session.call_tool(
                    "advance_time", {"days": 365}
                )
                _record(
                    _make_tool_output(
                        item.call_id, result=result.blocks[0].text
                    ),
                    result,
                )
                break

            try:
                tool_result = session.call_tool(
                    item.name,
                    json.loads(str(item.arguments)),
                )
            except Exception as e:
                error_text = f"Tool error: {e}"
                print(f"  -> {error_text[:200]}")
                tool_out = _make_tool_output(
                    item.call_id, error=error_text
                )
                input_list.append(tool_out)
                rollout.log_openai_response(tool_out)
                continue

            _record(
                _make_tool_output(
                    item.call_id, result=tool_result.blocks[0].text
                ),
                tool_result,
            )
            if finished:
                break

        if not had_function_call:
            nudge = {"role": "user", "content": _NUDGE}
            input_list.append(nudge)
            rollout.log_openai_response(nudge)

    or_client.rollout.close()

    return {
        "reward": reward,
        "finished": finished,
        "messages": input_list,
        "n_tool_calls": n_tool_calls,
    }


async def async_run_agent_loop(
    session,
    *,
    run_name: str = "brierbench-batch",
    rollout_name: str | None = None,
    split: str = "train",
    system_message: str | None = None,
    user_message: str | None = None,
    quiet: bool = False,
) -> dict:
    """Async agent loop for batch evaluation."""
    async_env = async_or_client.environments.get(
        name=_BRIERBENCH_ENV,
        base_url=_BRIERBENCH_URL,
    )
    tools = await async_env.list_tools(format="openai")

    if user_message is None:
        user_message = (await session.get_prompt())[0].text

    if rollout_name is None:
        rollout_name = f"{run_name}:{human_id()}"
    rollout = async_or_client.rollout.create(
        run_name=run_name,
        rollout_name=rollout_name,
        environment="DJRHails/brierbench",
        split=split,
        print_messages=not quiet,
    )

    input_list: list = []
    if system_message:
        sys_msg = {"role": "developer", "content": system_message}
        input_list.append(sys_msg)
        rollout.log_openai_response(sys_msg)
    input_list.append({"role": "user", "content": user_message})
    rollout.log_openai_response(input_list[-1])

    finished = False
    reward = None
    n_tool_calls = 0
    last_call_key: str | None = None
    repeat_count = 0

    def _record(tool_out: dict, tool_result) -> None:
        nonlocal reward, finished
        input_list.append(tool_out)
        rollout.log_openai_response(
            tool_out,
            reward=tool_result.reward,
            is_finished=tool_result.finished,
        )
        reward = tool_result.reward
        finished = tool_result.finished

    while not finished:
        response = await async_oai_client.responses.create(
            model=MODEL_NAME,
            tools=tools,
            input=input_list,
        )
        rollout.log_openai_response(response)
        input_list += response.output

        had_function_call = False
        for item in response.output:
            if item.type != "function_call":
                continue

            had_function_call = True
            n_tool_calls += 1
            call_key = f"{item.name}:{item.arguments}"

            if call_key == last_call_key:
                repeat_count += 1
            else:
                repeat_count = 0
            last_call_key = call_key

            if repeat_count >= _MAX_REPEATS:
                result = await session.call_tool(
                    "advance_time", {"days": 365}
                )
                _record(
                    _make_tool_output(
                        item.call_id, result=result.blocks[0].text
                    ),
                    result,
                )
                break

            try:
                tool_result = await session.call_tool(
                    item.name,
                    json.loads(str(item.arguments)),
                )
            except Exception as e:
                tool_out = _make_tool_output(
                    item.call_id, error=str(e)
                )
                input_list.append(tool_out)
                rollout.log_openai_response(tool_out)
                continue

            _record(
                _make_tool_output(
                    item.call_id, result=tool_result.blocks[0].text
                ),
                tool_result,
            )
            if finished:
                break

        if not had_function_call:
            nudge = {"role": "user", "content": _NUDGE}
            input_list.append(nudge)
            rollout.log_openai_response(nudge)

    return {
        "reward": reward,
        "finished": finished,
        "messages": input_list,
        "n_tool_calls": n_tool_calls,
    }
