"""Statistical helpers for final experiment analysis."""

from __future__ import annotations

from math import sqrt

from scipy.stats import binomtest


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int | str]:
    if total <= 0:
        return {"successes": successes, "total": total, "method": "Wilson 95%", "lower": float("nan"), "upper": float("nan")}
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return {
        "successes": successes,
        "total": total,
        "method": "Wilson 95%",
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
    }


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> dict[str, float | int | str]:
    if len(a_correct) != len(b_correct):
        return {"status": "NOT_AVAILABLE_FROM_ARTIFACTS", "reason": "unaligned lengths"}
    n01 = sum((not a) and b for a, b in zip(a_correct, b_correct))
    n10 = sum(a and (not b) for a, b in zip(a_correct, b_correct))
    discordant = n01 + n10
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(min(n01, n10), discordant, 0.5, alternative="two-sided").pvalue)
    return {
        "status": "COMPUTED",
        "method": "McNemar exact two-sided binomial",
        "n01_a_wrong_b_correct": n01,
        "n10_a_correct_b_wrong": n10,
        "discordant": discordant,
        "p_value": p_value,
    }

