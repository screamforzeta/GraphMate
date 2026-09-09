import pytest
import torch

from src.training.model_a.legal_mask import (
    apply_legal_mask,
    build_legal_mask_from_indices,
    legal_class_indices_from_fen,
    masked_cross_entropy,
)


def test_legal_class_indices_from_fen_uses_vocabulary_only():
    move_to_idx = {
        "e2e4": 0,
        "g1f3": 1,
        "a1a8": 2,
    }

    indices = legal_class_indices_from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move_to_idx,
    )

    assert indices == [0, 1]


def test_masked_loss_lets_legal_target_beat_illegal_argmax():
    logits = torch.tensor([[10.0, 1.0, 0.0]], requires_grad=True)
    targets = torch.tensor([1])
    legal_mask = build_legal_mask_from_indices([[1, 2]], num_classes=3)

    loss, masked_logits = masked_cross_entropy(logits, targets, legal_mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert masked_logits.argmax(dim=1).item() == 1
    assert logits.grad is not None


def test_multi_sample_masks_are_row_aligned():
    logits = torch.tensor(
        [
            [9.0, 1.0, 0.0],
            [9.0, 1.0, 0.0],
        ]
    )
    legal_mask = build_legal_mask_from_indices(
        [
            [1],
            [0],
        ],
        num_classes=3,
    )

    masked = apply_legal_mask(logits, legal_mask)

    assert masked.argmax(dim=1).tolist() == [1, 0]


def test_target_outside_mask_raises_clear_error():
    logits = torch.tensor([[0.0, 1.0]])
    targets = torch.tensor([0])
    legal_mask = build_legal_mask_from_indices([[1]], num_classes=2)

    with pytest.raises(ValueError, match="outside legal mask"):
        masked_cross_entropy(logits, targets, legal_mask)


def test_all_false_mask_raises_before_nan():
    logits = torch.tensor([[0.0, 1.0]])
    legal_mask = torch.zeros((1, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one"):
        apply_legal_mask(logits, legal_mask)


def test_mask_fill_value_is_finite_for_half_precision():
    logits = torch.tensor([[10.0, 1.0]], dtype=torch.float16)
    legal_mask = torch.tensor([[False, True]])

    masked = apply_legal_mask(logits, legal_mask)

    assert torch.isfinite(masked).all()
    assert masked.dtype == torch.float16
