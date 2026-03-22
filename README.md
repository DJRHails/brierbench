# BrierBench

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/DJRHails/brierbench)

## Description

BrierBench tests how well LLMs forecast real-world events. An agent receives a binary question from a prediction market or data source, searches the web for evidence, steps forward through simulated time, and submits probability estimates. Scores use a time-weighted Brier score: later predictions count more than earlier ones.

Questions come from [ForecastBench](https://www.forecastbench.org/) (Karger et al., 2025) and span Manifold, Metaculus, Polymarket, INFER, ACLED, FRED, Yahoo Finance, Wikipedia, and DBnomics.

## What it tests

- Calibrated probability estimation on real events
- Evidence gathering and synthesis via web search
- Temporal reasoning — updating beliefs as new information arrives
- Research planning under time pressure

## Compute

Network-isolated sandbox (`block_network=True`) for Python and bash. No GPU.

## Tasks

- **Train:** 6,671 tasks (Jul 2024 – Dec 2025)
- **Test:** 1,667 tasks (Dec 2025 – Mar 2026)

Each task gives the agent a question, background, resolution criteria, start date, and resolution date. The outcome is hidden.

## Scoring

ORS requires higher = better, but Brier scores are lower = better. The reward is flipped: `reward = 1 - time_weighted_brier_score`. To recover the raw score: `brier = 1 - reward`.

- Weight at time `t` is `t/T`, where `T` is the total horizon
- Later predictions carry more weight, rewarding refinement
- Unchanged predictions carry over between updates
- No prediction submitted → worst score (reward 0, Brier 1)
- Reward is continuous in [0, 1]

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search via Exa API; results filtered to content published on or before the simulated date |
| `bash` | Run a command in a network-isolated sandbox (python3, uv, numpy, pandas, scipy, etc. pre-installed) |
| `advance_time` | Move the simulated date forward by N days (1–365) |
| `submit_prediction` | Submit or update a probability estimate (0–1 exclusive) |
| `get_status` | Check current date, prediction, and time remaining |

## Time horizon

Multi-turn. Task horizons range from days to months. The agent chooses how many time steps to take and when to update.

## Secrets

- `EXA_API_KEY` for web search (not a well-known secret; pass as a tuple):
  ```python
  secrets = {
      "EXA_API_KEY": (exa_key, ["api.exa.ai"]),
      "OPENREWARD_API_KEY": openreward_key,
  }
  ```
- `OPENREWARD_API_KEY` for sandbox provisioning

## Contamination caveats

> **This benchmark does not guarantee a contamination-free evaluation.**

Two sources of leakage can inflate scores:

1. **Training cutoffs.** A model trained after a question's resolution date may already know the answer. A model with a January 2025 cutoff "knows" outcomes for questions that resolved in 2024, regardless of search results. There is no way to enforce a knowledge cutoff within the environment. Always report results alongside the model's training cutoff date.

2. **Search date filtering is imperfect.** Exa's `end_published_date` filters on publication-date metadata, which is not always accurate. Pages may be backdated, updated silently, or lack dates entirely. As a second line of defence, `web_search` runs each result through a cheap LLM (Gemini 2.0 Flash Lite) to check for references to events after the simulated date, and drops those that fail. The two-stage filter reduces but does not eliminate leakage; treat search results as best-effort.

For the strictest evaluation, use models whose training cutoff falls before the task start dates.

## Safety

- Sandbox has no network access (`block_network=True`)
- Web search is date-restricted (best-effort; see caveats above)
- No dual-use concerns; the domain is probability estimation on public events

## Citations

If you use BrierBench, please cite:

```bibtex
@misc{hails2025brierbench,
  author = {Daniel Hails},
  title  = {BrierBench: An OpenReward Environment for LLM Forecasting},
  year   = {2026},
  url    = {https://openreward.ai/DJRHails/brierbench},
}
```

BrierBench builds on questions from ForecastBench:

```bibtex
@inproceedings{karger2025forecastbench,
  title     = {ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities},
  author    = {Ezra Karger and Houtan Bastani and Chen Yueh-Han and Zachary Jacobs and Danny Halawi and Fred Zhang and Philip E. Tetlock},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
}
```
