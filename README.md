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

**Note:** ORS requires rewards where higher = better. The Brier score is
the opposite (lower = better), so the reward is flipped:
`reward = 1 - time_weighted_brier_score`. To recover the raw Brier score:
`brier = 1 - reward`.

- Predictions are weighted by `t/T` where `t` = days elapsed, `T` = total horizon
- Later predictions carry more weight, incentivising refinement over time
- Unchanged predictions carry over between updates
- No prediction submitted = worst score (reward 0, brier 1)
- Continuous reward in [0, 1] (brier in [0, 1])

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Web search via Exa API, restricted to content published on or before the current simulated date |
| `bash` | Execute a bash command in a network-isolated sandbox (python3, uv, numpy, pandas, scipy, etc. pre-installed) |
| `advance_time` | Advance the simulated date by N days (1--365) |
| `submit_prediction` | Submit or update probability estimate (0--1 exclusive) |
| `get_status` | Check current date, prediction, and time remaining |

## Time Horizon

Multi-turn. Typical task horizons range from days to months. Agents choose how many time steps to take and when to update predictions.

## Other Environment Requirements

- `EXA_API_KEY` secret for web search (not a well-known secret, pass as tuple):
  ```python
  secrets = {
      "EXA_API_KEY": (exa_key, ["api.exa.ai"]),
      "OPENREWARD_API_KEY": openreward_key,
  }
  ```
- `OPENREWARD_API_KEY` secret for sandbox provisioning

## Contamination Caveats

> **This benchmark does not guarantee a contamination-free evaluation.**

Two sources of information leakage can inflate scores:

1. **LLM training cutoffs.** Models trained after question resolution dates may have seen outcomes in their pre-training data. A model with a January 2025 cutoff already "knows" the answer to questions resolving in 2024, regardless of what `web_search` returns. There is no way to enforce a knowledge cutoff within the environment — this is an inherent limitation of evaluating forecasting with post-hoc models. Results should always be reported alongside the model's training cutoff date.

2. **Exa date filtering is imperfect.** The `end_published_date` parameter filters by *publication date metadata*, which is not always accurate. Pages may be backdated, updated after initial publication without changing the date, or lack publication dates entirely (in which case Exa may fall back to crawl date). This means `web_search` can occasionally return content containing information from after the simulated date. Treat search results as best-effort, not as a strict information boundary.

For the most rigorous evaluation, use models with training cutoffs *before* the task start dates, and treat Exa results as supplementary rather than authoritative.

## Safety

- Sandbox has no network access (`block_network=True`)
- Web search is date-restricted (best-effort; see caveats above)
- No dual-use risks; domain is probability estimation on public events

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
