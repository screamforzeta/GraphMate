"""Legal-move masking utilities for no-timing chess GAT ablations.

Purpose:
    Build per-sample legal move class masks from solver-position FEN strings
    and apply them to logits without changing the model architecture.
Input:
    PyG Data batches containing fen/y metadata and train-derived move_to_idx.
Output:
    Boolean legal masks [B, num_classes] and masked logits for CE/top-k.
Run:
    Imported by MODEL_A2_LEGAL_MASK_NO_TIMING training and tests.
"""

from __future__ import annotations

from functools import lru_cache

import chess
import torch


MASK_FILL_VALUE = -1.0e4


def legal_class_indices_from_fen(fen, move_to_idx):
    """Return vocabulary indices for legal UCI moves in a solver position."""

    return list(
        _cached_legal_class_indices(
            str(fen),
            tuple(sorted(move_to_idx.items())),
        )
    )


@lru_cache(maxsize=200000)
def _cached_legal_class_indices(fen, move_items):
    """Return cached legal class indices for one FEN/vocabulary signature."""

    move_to_idx = dict(move_items)
    board = chess.Board(str(fen))
    indices = []
    seen = set()
    for move in board.legal_moves:
        move_uci = move.uci()
        if move_uci not in move_to_idx:
            continue
        index = int(move_to_idx[move_uci])
        if index not in seen:
            seen.add(index)
            indices.append(index)
    return tuple(sorted(indices))


def build_legal_mask_from_indices(indices_per_sample, num_classes, device=None):
    """Build a dense boolean mask from compact legal class index lists."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    mask = torch.zeros(
        (len(indices_per_sample), num_classes),
        dtype=torch.bool,
        device=device,
    )
    for row, indices in enumerate(indices_per_sample):
        if not indices:
            raise ValueError(f"Legal mask row {row} has no legal classes.")
        for index in indices:
            if index < 0 or index >= num_classes:
                raise ValueError(f"Legal class index out of range: {index}")
        mask[row, torch.tensor(indices, dtype=torch.long, device=device)] = True
    return mask


def extract_batch_fens(batch):
    """Return graph-level FEN strings from a PyG batch."""

    fens = getattr(batch, "fen", None)
    if fens is None:
        raise ValueError("Batch is missing graph-level fen metadata.")
    if isinstance(fens, str):
        return [fens]
    return list(fens)


def build_legal_mask_for_batch(batch, move_to_idx, num_classes):
    """Build [B,num_classes] legal mask aligned with the PyG batch order."""

    indices_per_sample = [
        legal_class_indices_from_fen(fen, move_to_idx)
        for fen in extract_batch_fens(batch)
    ]
    return build_legal_mask_from_indices(
        indices_per_sample,
        num_classes,
        device=batch.y.device,
    )


def validate_targets_in_mask(targets, legal_mask):
    """Raise if any target class is not allowed by its row mask."""

    if legal_mask.ndim != 2:
        raise ValueError("legal_mask must have shape [batch_size, num_classes].")
    if targets.ndim != 1 or targets.shape[0] != legal_mask.shape[0]:
        raise ValueError("targets must align with legal_mask rows.")
    allowed = legal_mask[
        torch.arange(targets.shape[0], device=targets.device),
        targets,
    ]
    if not bool(allowed.all()):
        bad_rows = (~allowed).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Target class outside legal mask for rows: {bad_rows}")


def apply_legal_mask(logits, legal_mask, fill_value=MASK_FILL_VALUE):
    """Mask illegal logits with an AMP-safe finite negative value."""

    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must have the same shape.")
    if not bool(legal_mask.any(dim=1).all()):
        raise ValueError("Each legal mask row must allow at least one class.")
    return logits.masked_fill(~legal_mask, fill_value)


def masked_cross_entropy(logits, targets, legal_mask, criterion=None):
    """Compute CrossEntropyLoss after per-sample legal masking."""

    validate_targets_in_mask(targets, legal_mask)
    masked_logits = apply_legal_mask(logits, legal_mask)
    criterion = criterion or torch.nn.CrossEntropyLoss()
    loss = criterion(masked_logits, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite masked loss encountered.")
    return loss, masked_logits
