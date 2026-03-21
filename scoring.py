"""Time-weighted Brier scoring for BrierBench."""

from __future__ import annotations


def time_weighted_brier(
    predictions: list[tuple[int, float]],
    outcome: float,
    total_days: int,
) -> float:
    """Compute time-weighted Brier score.

    Args:
        predictions: List of (day_offset, probability) tuples, sorted
            by day_offset ascending. Each prediction carries over until
            the next update.
        outcome: Actual outcome (0.0 or 1.0).
        total_days: Total horizon T in days.

    Returns:
        Weighted Brier score in [0, 1].  Lower is better.
    """
    if not predictions or total_days <= 0:
        return 1.0

    weight_sum = 0.0
    score_sum = 0.0

    for i, (day, prob) in enumerate(predictions):
        # Active from this day up to (but not including) next update,
        # or through total_days (inclusive) for the last prediction.
        if i + 1 < len(predictions):
            end_exclusive = predictions[i + 1][0]
        else:
            end_exclusive = total_days + 1

        brier = (prob - outcome) ** 2
        for d in range(day, end_exclusive):
            w = d / total_days
            weight_sum += w
            score_sum += w * brier

    if weight_sum == 0:
        return 1.0

    return score_sum / weight_sum
