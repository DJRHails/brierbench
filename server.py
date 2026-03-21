"""BrierBench: an OpenReward environment for evaluating LLM forecasting.

Agents receive a binary forecasting question and must predict the probability
of resolution.  They can search the web (date-restricted via Exa), run
Python/Bash in a network-isolated sandbox, advance simulated time, and
submit/update their prediction at any point.

Scoring uses a *time-weighted* Brier score: later predictions carry more
weight (weight = t/T where t = days elapsed, T = total horizon).  If the
prediction is unchanged between updates it carries over at each timestep.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from typing import ClassVar

from exa_py import Exa
from pydantic import BaseModel, Field

from openreward import AsyncOpenReward, SandboxSettings
from openreward.environments import (
    Environment,
    Server,
    Split,
    TextBlock,
    ToolOutput,
    tool,
)

from scoring import time_weighted_brier

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_DIR = os.environ.get("BRIERBENCH_DATA_DIR", "/orwd_data")


def _load_tasks_file(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


_train_tasks: list[dict] | None = None
_test_tasks: list[dict] | None = None


def _get_tasks(split: str) -> list[dict]:
    global _train_tasks, _test_tasks  # noqa: PLW0603
    if split == "train":
        if _train_tasks is None:
            _train_tasks = _load_tasks_file(f"{DATA_DIR}/train.json")
        return _train_tasks
    if split == "test":
        if _test_tasks is None:
            _test_tasks = _load_tasks_file(f"{DATA_DIR}/test.json")
        return _test_tasks
    raise ValueError(f"Unknown split: {split}")


# ---------------------------------------------------------------------------
# Tool parameter models
# ---------------------------------------------------------------------------


class ExaSearchParams(BaseModel):
    """Parameters for web search via Exa."""

    query: str = Field(
        ..., description="Natural-language search query."
    )
    num_results: int = Field(
        5, description="Number of results to return (1-10).", ge=1, le=10
    )


class BashParams(BaseModel):
    """Parameters for bash command execution."""

    command: str = Field(
        ..., description="Bash command to execute in the sandbox."
    )


class PythonParams(BaseModel):
    """Parameters for Python code execution."""

    code: str = Field(
        ..., description="Python code to execute in the sandbox."
    )


class AdvanceTimeParams(BaseModel):
    """Parameters for advancing simulated time."""

    days: int = Field(
        ...,
        description="Number of days to advance (1-365).",
        ge=1,
        le=365,
    )


class SubmitPredictionParams(BaseModel):
    """Parameters for submitting or updating a prediction."""

    probability: float = Field(
        ...,
        description=(
            "Predicted probability that the question resolves YES "
            "(between 0 and 1 exclusive)."
        ),
        gt=0.0,
        lt=1.0,
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BrierBench(Environment):
    """Forecasting environment with time-weighted Brier scoring."""

    toolsets: ClassVar[list] = []

    def __init__(
        self,
        task_spec: dict = {},
        secrets: dict[str, str] = {},
    ) -> None:
        super().__init__(task_spec, secrets=secrets)

        # Parse task
        self.task_id: str = task_spec["id"]
        self.question_text: str = task_spec["question"]
        self.background: str = task_spec.get("background", "")
        self.resolution_criteria: str = task_spec.get(
            "resolution_criteria", ""
        )
        self.source: str = task_spec.get("source", "")

        self.start_date = date.fromisoformat(task_spec["start_date"])
        self.resolution_date = date.fromisoformat(
            task_spec["resolution_date"]
        )
        self.outcome: float = float(task_spec["outcome"])

        self.total_days = (self.resolution_date - self.start_date).days

        # Simulated clock
        self.current_date: date = self.start_date

        # Prediction log: list of (day_offset, probability)
        self.predictions: list[tuple[int, float]] = []

        # Exa client (initialised in setup)
        self._exa: Exa | None = None
        self._exa_api_key = secrets.get("EXA_API_KEY", "")

        # Sandbox client
        self._or_client = AsyncOpenReward(
            api_key=secrets.get("OPENREWARD_API_KEY", ""),
        )
        self._sandbox_settings = SandboxSettings(
            environment="DJRHails/brierbench",
            image="python:3.11-slim",
            machine="0.5:0.5",
            block_network=True,
        )
        self._sandbox = self._or_client.sandbox(self._sandbox_settings)

    async def setup(self) -> None:
        """Start sandbox and initialise Exa client."""
        if self._exa_api_key:
            self._exa = Exa(api_key=self._exa_api_key)
        await self._sandbox.start()

    async def teardown(self) -> None:
        """Stop sandbox."""
        await self._sandbox.stop()

    # -- Class methods -------------------------------------------------------

    @classmethod
    def list_splits(cls) -> list[Split]:
        return [
            Split(name="train", type="train"),
            Split(name="test", type="test"),
        ]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict]:
        return _get_tasks(split)

    def get_prompt(self) -> list[TextBlock]:
        lines = [
            "You are an expert forecaster. Your task is to predict the "
            "probability that the following question resolves YES.",
            "",
            f"**Question:** {self.question_text}",
        ]
        if self.background:
            lines += ["", f"**Background:** {self.background}"]
        if self.resolution_criteria:
            lines += [
                "",
                f"**Resolution Criteria:** {self.resolution_criteria}",
            ]
        lines += [
            "",
            f"**Current Date:** {self.current_date.isoformat()}",
            f"**Resolution Date:** {self.resolution_date.isoformat()}",
            f"**Days Remaining:** {self._days_remaining()}",
            "",
            "## Available Tools",
            "",
            "- `exa_search`: Search the web for information "
            "(results restricted to content published on or before "
            "the current simulated date).",
            "- `python`: Execute Python code in a sandboxed environment "
            "(no network access).",
            "- `bash`: Execute a bash command in a sandboxed environment "
            "(no network access).",
            "- `advance_time`: Advance the simulated date by N days. "
            "New information becomes available via exa_search after "
            "advancing.",
            "- `submit_prediction`: Submit or update your probability "
            "estimate (0-1 exclusive). You can update at any time. "
            "Later predictions carry more weight.",
            "- `get_status`: Check current date, prediction, and "
            "time remaining.",
            "",
            "## Scoring",
            "",
            "Your reward is based on a time-weighted Brier score. "
            "Predictions made later carry more weight (weight = t/T "
            "where t = days elapsed, T = total horizon). "
            "Submit early and refine as you gather more information.",
        ]
        return [TextBlock(type="text", text="\n".join(lines))]

    # -- Helpers -------------------------------------------------------------

    def _days_elapsed(self) -> int:
        return (self.current_date - self.start_date).days

    def _days_remaining(self) -> int:
        return (self.resolution_date - self.current_date).days

    def _is_resolved(self) -> bool:
        return self.current_date >= self.resolution_date

    def _current_prediction(self) -> float | None:
        if not self.predictions:
            return None
        return self.predictions[-1][1]

    def _compute_reward(self) -> float:
        """Reward = 1 - time_weighted_brier.  Higher is better."""
        bs = time_weighted_brier(
            self.predictions, self.outcome, self.total_days
        )
        return 1.0 - bs

    def _status_text(self) -> str:
        pred = self._current_prediction()
        pred_str = f"{pred:.4f}" if pred is not None else "None"
        return (
            f"Date: {self.current_date.isoformat()} | "
            f"Days elapsed: {self._days_elapsed()}/{self.total_days} | "
            f"Days remaining: {self._days_remaining()} | "
            f"Current prediction: {pred_str} | "
            f"Updates so far: {len(self.predictions)}"
        )

    # -- Tools ---------------------------------------------------------------

    @tool
    async def exa_search(self, params: ExaSearchParams) -> ToolOutput:
        """Search the web via Exa, restricted to the current simulated date."""
        if self._exa is None:
            return ToolOutput(
                blocks=[
                    TextBlock(
                        type="text",
                        text="Error: EXA_API_KEY not provided.",
                    )
                ],
            )

        end_date = (self.current_date + timedelta(days=1)).isoformat()
        try:
            response = self._exa.search_and_contents(
                params.query,
                num_results=params.num_results,
                end_published_date=end_date,
                type="auto",
                text={"max_characters": 3000},
            )
        except Exception as e:
            return ToolOutput(
                blocks=[
                    TextBlock(type="text", text=f"Exa search error: {e}")
                ],
            )

        results = []
        for r in response.results:
            entry = f"### {r.title}\n**URL:** {r.url}\n"
            if r.published_date:
                entry += f"**Published:** {r.published_date}\n"
            if r.text:
                entry += f"\n{r.text[:2000]}\n"
            results.append(entry)

        if not results:
            text = "No results found."
        else:
            text = (
                f"Found {len(results)} results "
                f"(published on or before {self.current_date.isoformat()}):"
                f"\n\n{'---'.join(results)}"
            )

        return ToolOutput(
            blocks=[TextBlock(type="text", text=text)],
        )

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """Execute a bash command in the network-isolated sandbox."""
        try:
            output = await self._sandbox.check_run(params.command)
            text = str(output) if output else "(no output)"
        except Exception as e:
            text = f"Error: {e}"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=text)],
        )

    @tool
    async def python(self, params: PythonParams) -> ToolOutput:
        """Execute Python code in the network-isolated sandbox."""
        escaped_code = params.code.replace("'", "'\\''")
        command = f"python3 -c '{escaped_code}'"
        try:
            output = await self._sandbox.check_run(command)
            text = str(output) if output else "(no output)"
        except Exception as e:
            text = f"Error: {e}"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=text)],
        )

    @tool
    async def advance_time(self, params: AdvanceTimeParams) -> ToolOutput:
        """Advance the simulated date by N days."""
        if self._is_resolved():
            reward = self._compute_reward()
            return ToolOutput(
                blocks=[
                    TextBlock(
                        type="text",
                        text=(
                            "The question has already resolved. "
                            f"Final reward: {reward:.4f}"
                        ),
                    )
                ],
                reward=reward,
                finished=True,
            )

        new_date = self.current_date + timedelta(days=params.days)

        if new_date >= self.resolution_date:
            self.current_date = self.resolution_date
            reward = self._compute_reward()
            return ToolOutput(
                blocks=[
                    TextBlock(
                        type="text",
                        text=(
                            f"Advanced to resolution date "
                            f"{self.resolution_date.isoformat()}. "
                            f"The question has now resolved.\n\n"
                            f"Outcome: {'YES' if self.outcome == 1.0 else 'NO'}\n"
                            f"Your final prediction: "
                            f"{self._current_prediction()}\n"
                            f"Reward: {reward:.4f}"
                        ),
                    )
                ],
                reward=reward,
                finished=True,
            )

        self.current_date = new_date
        return ToolOutput(
            blocks=[
                TextBlock(
                    type="text",
                    text=(
                        f"Advanced {params.days} day(s). "
                        f"{self._status_text()}"
                    ),
                )
            ],
        )

    @tool
    async def submit_prediction(
        self, params: SubmitPredictionParams
    ) -> ToolOutput:
        """Submit or update a probability prediction."""
        day_offset = self._days_elapsed()
        self.predictions.append((day_offset, params.probability))

        if self._is_resolved():
            reward = self._compute_reward()
            return ToolOutput(
                blocks=[
                    TextBlock(
                        type="text",
                        text=(
                            f"Prediction {params.probability:.4f} recorded "
                            f"at day {day_offset}. Question already resolved."
                            f"\n\nOutcome: "
                            f"{'YES' if self.outcome == 1.0 else 'NO'}\n"
                            f"Reward: {reward:.4f}"
                        ),
                    )
                ],
                reward=reward,
                finished=True,
            )

        return ToolOutput(
            blocks=[
                TextBlock(
                    type="text",
                    text=(
                        f"Prediction updated to {params.probability:.4f} "
                        f"at day {day_offset}/{self.total_days}.\n"
                        f"{self._status_text()}"
                    ),
                )
            ],
        )

    @tool
    async def get_status(self, params: dict = {}) -> ToolOutput:
        """Get current status: date, prediction, time remaining."""
        return ToolOutput(
            blocks=[
                TextBlock(type="text", text=self._status_text())
            ],
        )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

Server([BrierBench]).run()
