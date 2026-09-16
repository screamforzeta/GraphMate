"""Two-stage inference helper for MODEL A4 post-move reranking.

Purpose:
    Run frozen A3 retrieval followed by A4 post-move reranking for one FEN.
Input:
    Solver-position FEN, frozen A3 checkpoint, and A4 checkpoint.
Output:
    A3 Top-K and final A4 ranking with scores.
Run:
    Imported by CLI/evaluation code; does not use TargetMove.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Batch

from src.graph.graph_builder import build_graph
from src.graph.pyg_dataset import load_move_encoder
from src.training.model_a.model_a3_legal_scorer import legal_candidates_from_fen
from src.training.model_a.model_a4_postmove_reranker import (
    ModelA4PostMoveConfig,
    build_model_a4,
    build_postmove_graph,
    load_frozen_a3,
    select_topk_a3,
    _a3_candidate_features,
)


def load_a4_for_inference(checkpoint_path, device):
    """Load an A4 checkpoint for inference.

    Parameters:
        checkpoint_path: Path to artifacts/model_a4_postmove_gnn_reranker/best.pt.
        device: Torch device.
    Returns:
        Tuple (model, checkpoint).
    Side effects:
        Reads checkpoint_path.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    config_dict = checkpoint.get("config", {})
    config = ModelA4PostMoveConfig(**{
        key: value
        for key, value in config_dict.items()
        if key in ModelA4PostMoveConfig.__dataclass_fields__
    })
    model = build_model_a4(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_a4_from_fen(
    fen,
    a3_checkpoint,
    a4_checkpoint,
    device="cpu",
    top_k=5,
    amp=False,
):
    """Run A4 two-stage inference for one solver-position FEN.

    Parameters:
        fen: Solver-position FEN.
        a3_checkpoint: Frozen A3 checkpoint path.
        a4_checkpoint: A4 checkpoint path.
        device: Torch device string or object.
        top_k: A3 retrieval size.
        amp: Enable CUDA autocast.
    Returns:
        Dict with A3 Top-K, A4 ranking, and top move.
    Side effects:
        Loads checkpoints and move vocabulary.
    """

    device = torch.device(device)
    move_to_idx = load_move_encoder()
    dummy_target = next(iter(move_to_idx))
    original_graph = build_graph(fen, dummy_target, move_to_idx)
    batch = Batch.from_data_list([original_graph]).to(device)
    a3, _ = load_frozen_a3(a3_checkpoint, device)
    a4, _ = load_a4_for_inference(a4_checkpoint, device)
    use_amp = bool(amp and device.type == "cuda")
    with torch.no_grad(), torch.amp.autocast(device.type, enabled=use_amp):
        candidate_moves = [legal_candidates_from_fen(fen)]
        output = a3(batch, candidate_moves)
        a3_features = _a3_candidate_features(a3, batch, candidate_moves)
        placeholder_target = torch.tensor([0], dtype=torch.long, device=device)
        topk = select_topk_a3(output, placeholder_target, top_k)[0]
        selected_flat = topk["local_indices"]
        postmove_graphs = [
            build_postmove_graph(fen, move_uci, dummy_target, move_to_idx)
            for move_uci in topk["uci"]
        ]
        postmove_batch = Batch.from_data_list(postmove_graphs).to(device)
        score_features = torch.tensor(
            list(zip(topk["raw_scores"], topk["centered_scores"])),
            dtype=torch.float,
            device=device,
        )
        scores = a4(
            a3_features[selected_flat],
            postmove_batch,
            score_features,
        )
        ordered = torch.argsort(scores, descending=True)
    ranking = [
        {
            "uci": topk["uci"][int(index.item())],
            "a3_score": topk["raw_scores"][int(index.item())],
            "a4_score": float(scores[int(index.item())].detach().cpu().item()),
        }
        for index in ordered
    ]
    return {
        "a3_topk": [
            {"uci": uci, "a3_score": score}
            for uci, score in zip(topk["uci"], topk["raw_scores"])
        ],
        "a4_ranking": ranking,
        "a4_top1": ranking[0]["uci"] if ranking else None,
    }
