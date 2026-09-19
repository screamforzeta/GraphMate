"""Canonical prompt templates for local LLM chess benchmarks."""

from __future__ import annotations

import hashlib


PROMPT_TEMPLATE_VERSION = "llm_chess_uci_v1"
SYSTEM_PROMPT = (
    "You are solving a chess position.\n"
    "Return exactly one legal chess move for the side to move.\n"
    "Do not provide explanation or analysis."
)
USER_TEMPLATE = "FEN:\n{fen}\n\nReturn your move in UCI notation only."


def build_prompt(fen):
    """Build the canonical no-leakage prompt for one FEN."""

    return {
        "system": SYSTEM_PROMPT,
        "user": USER_TEMPLATE.format(fen=fen),
        "version": PROMPT_TEMPLATE_VERSION,
    }


def prompt_hash():
    """Return a stable hash of the frozen prompt template."""

    payload = f"{PROMPT_TEMPLATE_VERSION}\n{SYSTEM_PROMPT}\n{USER_TEMPLATE}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

