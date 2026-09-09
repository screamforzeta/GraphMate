"""Metrics for graph-level chess move classification.

Purpose:
    Provide small, reusable metrics for model training and evaluation.
Input:
    Raw model logits and integer graph-level targets.
Output:
    Accuracy values in the range [0.0, 1.0].
Role:
    Keeps metric code independent from trainer and CLI concerns.
"""

import torch


def compute_topk_accuracies(logits, targets, top_k=(1, 3, 5)):
    """Compute top-k accuracies from raw logits.

    Parameters:
        logits: Tensor with shape [batch_size, num_classes].
        targets: Long tensor with shape [batch_size].
        top_k: Iterable of k values to evaluate.
    Returns:
        Dict mapping each requested k to an accuracy in [0.0, 1.0].
    Side effects:
        None.
    """

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape [batch_size, num_classes]."
        )
    if targets.ndim != 1:
        raise ValueError(
            "targets must have shape [batch_size]."
        )
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "logits and targets must have the same batch size."
        )

    num_examples = targets.numel()
    if num_examples == 0:
        return {
            k: 0.0
            for k in top_k
        }

    num_classes = logits.shape[1]
    accuracies = {}

    for k in top_k:
        safe_k = min(
            int(k),
            num_classes,
        )
        topk = logits.topk(
            safe_k,
            dim=1,
        ).indices
        correct = (
            topk == targets.view(-1, 1)
        ).any(dim=1)
        accuracies[k] = (
            correct.float().mean().item()
        )

    return accuracies
