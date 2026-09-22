"""Scoring helpers for external classic held-out chess problems."""

from __future__ import annotations


def is_accepted_classic_key(predicted_uci: str, accepted_key_moves_uci: list[str]) -> bool:
    """Return whether a predicted UCI move is accepted for classic held-out scoring."""

    return str(predicted_uci).strip() in set(accepted_key_moves_uci)

