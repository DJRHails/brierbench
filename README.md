# BrierBench

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/DJRHails/brierbench)

## Description

BrierBench evaluates LLM forecasting accuracy on real-world binary questions from prediction markets and data sources. Agents receive a question, search the web for information, advance through simulated time, and submit probability predictions scored with a time-weighted Brier score.

Questions are sourced from [ForecastBench](https://www.forecastbench.org/) (Karger et al., 2025), covering Manifold, Metaculus, Polymarket, INFER, ACLED, FRED, Yahoo Finance, Wikipedia, and DBnomics.

## Capabilities

- Calibrated probability estimation on real-world events
- Information gathering and synthesis via web search
- Temporal reasoning (updating beliefs as new information arrives)
- Research planning under time constraints

## Compute Requirements

Sandbox with network isolation (`block_network=True`) for Python/bash execution. No GPU required.

## Tasks

- **Train:** 6,671 tasks (Jul 2024 -- Dec 2025)
- **Test:** 1,667 tasks (Dec 2025 -- Mar 2026)

Each task provides a question, background, resolution criteria, start date, and resolution date. The agent does not see the outcome.

## Reward Structure

Time-weighted Brier score: `reward = 1 - weighted_brier`.

- Predictions are weighted by `t/T` where `t` = days elapsed, `T` = total horizon
- Later predictions carry more weight, incentivising refinement over time
- Unchanged predictions carry over between updates
- No prediction submitted = worst score (reward 0)
- Continuous reward in [0, 1]

## Tools

| Tool | Description |
|------|-------------|
| `exa_search` | Web search via Exa API, restricted to content published on or before the current simulated date |
| `python` | Execute Python code in a network-isolated sandbox |
| `bash` | Execute a bash command in a network-isolated sandbox |
| `advance_time` | Advance the simulated date by N days (1--365) |
| `submit_prediction` | Submit or update probability estimate (0--1 exclusive) |
| `get_status` | Check current date, prediction, and time remaining |

## Time Horizon

Multi-turn. Typical task horizons range from days to months. Agents choose how many time steps to take and when to update predictions.

## Other Environment Requirements

- `EXA_API_KEY` secret for web search
- `OPENREWARD_API_KEY` secret for sandbox provisioning

## Safety

- Sandbox has no network access (`block_network=True`)
- Web search is date-restricted to prevent future information leakage
- No dual-use risks; domain is probability estimation on public events

## Citations

```bibtex
@article{karger2025forecastbench,
  author  = {Karger, Ezra and others},
  title   = {ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities},
  year    = {2025},
  journal = {arXiv preprint},
}
```
