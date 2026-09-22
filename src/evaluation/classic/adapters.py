"""Model adapter contracts for classic held-out evaluation."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import torch
from torch_geometric.data import Batch

from src.evaluation.model_a.model_a3_vs_a4_terminal import MODEL_A4_CHECKPOINT
from src.evaluation.model_a.model_a_vs_a2_vs_a3 import MODEL_A3_CHECKPOINT, load_eval_model_a3
from src.graph.graph_builder import build_graph
from src.graph.pyg_dataset import load_move_encoder
from src.llm.parsing import PARSER_VERSION, parse_uci_response
from src.llm.prompting import PROMPT_TEMPLATE_VERSION, build_prompt, prompt_hash
from src.llm.model_registry import LLM_BENCHMARK_MODELS, generation_config_for_model
from src.llm.ollama_client import OllamaClient, resolve_endpoint
from src.training.model_a.model_a4_postmove_reranker import (
    _a3_candidate_features,
    build_postmove_graph,
    select_topk_a3,
)
from src.inference.model_a.postmove_reranker import load_a4_for_inference
from src.training.model_a.model_a3_legal_scorer import legal_candidates_from_fen


A3_CHECKPOINT_PATH = MODEL_A3_CHECKPOINT
A4_CHECKPOINT_PATH = MODEL_A4_CHECKPOINT
B_CHECKPOINT_PATH = Path("artifacts/model_b_timing_legal_move_scorer/best.pt")
MODEL_B_CLASSIC_TIMING_PROTOCOL = "not_applicable_timing_unavailable"


def sha256_optional(path: Path) -> str | None:
    """Return SHA256 for an existing artifact, otherwise None."""

    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class ClassicModelAdapter:
    """Small interface implemented by classic benchmark systems."""

    model_id: str
    model_family: str
    model_version: str

    def prepare(self) -> None:
        """Prepare resources before prediction."""

    def predict(self, sample):
        """Return a prediction payload for one sample."""

        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        """Return identity metadata for run config and prediction records."""

        return {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "model_version": self.model_version,
        }


class A3ClassicAdapter(ClassicModelAdapter):
    """Adapter contract for frozen A3 legal-candidate scoring."""

    def __init__(self, checkpoint_path: Path = A3_CHECKPOINT_PATH, device: str = "cpu", amp: bool = False):
        super().__init__("MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING", "gnn_a3", "frozen")
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.amp = bool(amp)
        self.model = None
        self.checkpoint = None
        self.move_to_idx = None

    def prepare(self) -> None:
        """Load the frozen A3 checkpoint and move encoder."""

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"A3 checkpoint not found: {self.checkpoint_path}")
        self.model, self.checkpoint = load_eval_model_a3(self.checkpoint_path, self.device)
        self.model.eval()
        self.move_to_idx = load_move_encoder()

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload.update(
            {
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_sha256": sha256_optional(self.checkpoint_path),
                "candidate_source": "python_chess_legal_moves_from_canonical_fen",
                "applies_lichess_moves0": False,
                "requires_global_1786_classifier": False,
                "device": str(self.device),
                "amp": self.amp,
            }
        )
        return payload

    def candidates_for_sample(self, sample) -> list[str]:
        """Return legal UCI candidates from the canonical classic FEN."""

        return [move.uci() for move in legal_candidates_from_fen(sample.fen)]

    def predict(self, sample):
        """Run frozen A3 inference on one canonical classic FEN."""

        if self.model is None or self.move_to_idx is None:
            self.prepare()
        started = time.perf_counter()
        dummy_target = next(iter(self.move_to_idx))
        graph = build_graph(sample.fen, dummy_target, self.move_to_idx)
        batch = Batch.from_data_list([graph]).to(self.device)
        candidate_moves = [legal_candidates_from_fen(sample.fen)]
        use_amp = bool(self.amp and self.device.type == "cuda")
        with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=use_amp):
            output = self.model(batch, candidate_moves)
            scores = output["scores"]
            ordered = torch.argsort(scores, descending=True)
        ranked = [output["candidate_uci"][int(index.item())] for index in ordered]
        raw_scores = [float(scores[int(index.item())].detach().cpu().item()) for index in ordered]
        return {
            "predicted_move_uci": ranked[0] if ranked else None,
            "ranked_moves_uci": ranked,
            "is_legal": bool(ranked),
            "runtime_seconds": time.perf_counter() - started,
            "diagnostics": {
                "candidate_count": len(ranked),
                "scores_by_rank": raw_scores,
                "accepted_keys_used_for_inference": False,
                "source_key_used_for_inference": False,
            },
        }


class A4ClassicAdapter(ClassicModelAdapter):
    """Adapter contract for frozen A4 A3-Top5 plus post-move reranking."""

    def __init__(
        self,
        a3_checkpoint_path: Path = A3_CHECKPOINT_PATH,
        a4_checkpoint_path: Path = A4_CHECKPOINT_PATH,
        top_k: int = 5,
        device: str = "cpu",
        amp: bool = False,
    ):
        super().__init__("MODEL_A4_POSTMOVE_GNN_RERANKER_NO_TIMING", "gnn_a4", "frozen")
        self.a3_checkpoint_path = Path(a3_checkpoint_path)
        self.a4_checkpoint_path = Path(a4_checkpoint_path)
        self.top_k = int(top_k)
        self.device = torch.device(device)
        self.amp = bool(amp)
        self.a3 = None
        self.a4 = None
        self.move_to_idx = None

    def prepare(self) -> None:
        """Load frozen A3 retrieval and A4 reranker checkpoints."""

        if not self.a3_checkpoint_path.exists():
            raise FileNotFoundError(f"A3 retrieval checkpoint not found: {self.a3_checkpoint_path}")
        if not self.a4_checkpoint_path.exists():
            raise FileNotFoundError(f"A4 checkpoint not found: {self.a4_checkpoint_path}")
        self.a3, _ = load_eval_model_a3(self.a3_checkpoint_path, self.device)
        self.a3.eval()
        for parameter in self.a3.parameters():
            parameter.requires_grad = False
        self.a4, _ = load_a4_for_inference(self.a4_checkpoint_path, self.device)
        self.a4.eval()
        self.move_to_idx = load_move_encoder()

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload.update(
            {
                "a3_checkpoint_path": str(self.a3_checkpoint_path),
                "a3_checkpoint_sha256": sha256_optional(self.a3_checkpoint_path),
                "a4_checkpoint_path": str(self.a4_checkpoint_path),
                "a4_checkpoint_sha256": sha256_optional(self.a4_checkpoint_path),
                "retrieval_model": "frozen_A3",
                "retrieval_top_k": self.top_k,
                "retrieval_uses_target_or_accepted_keys": False,
                "postmove_graph_source": "apply_candidate_to_canonical_fen",
                "device": str(self.device),
                "amp": self.amp,
            }
        )
        return payload

    def retrieval_inputs(self, sample) -> dict[str, Any]:
        """Return target-free retrieval inputs for audit and tests."""

        return {"fen": sample.fen, "accepted_key_moves_uci": None, "source_key_move_uci": None}

    def predict(self, sample):
        """Run frozen A4 inference with genuine A3 Top-K retrieval."""

        if self.a3 is None or self.a4 is None or self.move_to_idx is None:
            self.prepare()
        started = time.perf_counter()
        dummy_target = next(iter(self.move_to_idx))
        graph = build_graph(sample.fen, dummy_target, self.move_to_idx)
        batch = Batch.from_data_list([graph]).to(self.device)
        candidate_moves = [legal_candidates_from_fen(sample.fen)]
        placeholder_target = torch.tensor([0], dtype=torch.long, device=self.device)
        use_amp = bool(self.amp and self.device.type == "cuda")
        with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=use_amp):
            a3_output = self.a3(batch, candidate_moves)
            a3_features = _a3_candidate_features(self.a3, batch, candidate_moves)
            topk = select_topk_a3(a3_output, placeholder_target, self.top_k)[0]
            selected_flat = topk["local_indices"]
            postmove_graphs = [
                build_postmove_graph(sample.fen, move_uci, dummy_target, self.move_to_idx)
                for move_uci in topk["uci"]
            ]
            postmove_batch = Batch.from_data_list(postmove_graphs).to(self.device)
            score_features = torch.tensor(
                list(zip(topk["raw_scores"], topk["centered_scores"])),
                dtype=torch.float,
                device=self.device,
            )
            rerank_scores = self.a4(a3_features[selected_flat], postmove_batch, score_features)
            ordered = torch.argsort(rerank_scores, descending=True)
        ranked = [topk["uci"][int(index.item())] for index in ordered]
        accepted = set(sample.accepted_key_moves_uci)
        retrieval_success = bool(accepted.intersection(topk["uci"]))
        a4_correct = bool(ranked and ranked[0] in accepted)
        return {
            "predicted_move_uci": ranked[0] if ranked else None,
            "ranked_moves_uci": ranked,
            "is_legal": bool(ranked),
            "runtime_seconds": time.perf_counter() - started,
            "diagnostics": {
                "a3_topk": topk["uci"],
                "a3_raw_scores": topk["raw_scores"],
                "a3_centered_scores": topk["centered_scores"],
                "a4_scores_by_rank": [
                    float(rerank_scores[int(index.item())].detach().cpu().item())
                    for index in ordered
                ],
                "retrieval_success": retrieval_success,
                "reranking_failure": bool(retrieval_success and not a4_correct),
                "retrieval_uses_target_or_accepted_keys": False,
                "reranker_uses_target_or_accepted_keys": False,
            },
        }


class ModelBClassicAdapter(ClassicModelAdapter):
    """Adapter contract for Model B with explicit timing compatibility state."""

    def __init__(self, checkpoint_path: Path = B_CHECKPOINT_PATH):
        super().__init__("MODEL_B_TIMING_LEGAL_MOVE_SCORER", "gnn_b_timing", "frozen")
        self.checkpoint_path = Path(checkpoint_path)
        self.timing_protocol = MODEL_B_CLASSIC_TIMING_PROTOCOL

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload.update(
            {
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_sha256": sha256_optional(self.checkpoint_path),
                "timing_data_available": False,
                "timing_protocol": self.timing_protocol,
                "timing_required_fields": [
                    "previous_move_time",
                    "original_move_time",
                    "time_is_synthetic",
                ],
                "timing_interpretation": "Model B requires Lichess-derived timing fields that YACPDB classic positions do not provide.",
                "random_timing_generation": False,
            }
        )
        return payload

    def timing_values_for_sample(self, sample) -> None:
        """Return no timing values; classic Model B is not applicable by default."""

        return None

    def prepare(self) -> None:
        """Fail closed because the frozen classic protocol has no timing inputs."""

        raise RuntimeError(
            "Model B is not applicable to yacpdb_classic_v1: "
            "the frozen classic protocol defines no position-specific timing inputs."
        )


class QwenClassicAdapter(ClassicModelAdapter):
    """Adapter contract for frozen Qwen/Ollama LLM protocol."""

    def __init__(self, ollama_model_id: str, endpoint: str | None = None, timeout: int = 120):
        super().__init__(ollama_model_id, "llm_qwen", "frozen_protocol")
        self.ollama_model_id = ollama_model_id
        self.endpoint = resolve_endpoint(endpoint)
        self.timeout = timeout
        self.client = OllamaClient(self.endpoint, timeout=timeout)
        self.registry_id = self._registry_id_for_ollama_model(ollama_model_id)

    @staticmethod
    def _registry_id_for_ollama_model(ollama_model_id: str) -> str:
        for registry_id, entry in LLM_BENCHMARK_MODELS.items():
            if entry["ollama_model"] == ollama_model_id:
                return registry_id
        raise ValueError(f"Unsupported frozen Qwen model for classic benchmark: {ollama_model_id}")

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload.update(
            {
                "ollama_model_id": self.ollama_model_id,
                "prompt_version": PROMPT_TEMPLATE_VERSION,
                "prompt_hash": prompt_hash(),
                "parser_version": PARSER_VERSION,
                "target_leakage": False,
                "model_digest": LLM_BENCHMARK_MODELS[self.registry_id]["digest"],
                "endpoint": self.endpoint,
                "generation_config": generation_config_for_model(self.registry_id),
            }
        )
        return payload

    def build_prompt_for_sample(self, sample) -> dict[str, str]:
        """Build the frozen LLM prompt from canonical FEN only."""

        return build_prompt(sample.fen)

    def parse_response_for_sample(self, sample, raw_response: str):
        """Parse an LLM response with the frozen strict UCI parser."""

        return parse_uci_response(raw_response, sample.fen)

    def predict(self, sample):
        """Run one frozen Qwen/Ollama prediction for a classic sample."""

        prompt = self.build_prompt_for_sample(sample)
        config = generation_config_for_model(self.registry_id)
        started = time.perf_counter()
        result = self.client.generate(
            model=self.ollama_model_id,
            prompt=prompt["user"],
            system=prompt["system"],
            options=config["options"],
            think=config["think"],
        )
        runtime = time.perf_counter() - started
        if result.get("error"):
            return {
                "predicted_move_uci": None,
                "parse_success": False,
                "is_legal": False,
                "runtime_seconds": runtime,
                "diagnostics": {
                    "outcome_category": "runtime_failure",
                    "error": result["error"],
                    "raw_model_response": result.get("response", ""),
                },
            }
        parsed = self.parse_response_for_sample(sample, result.get("response", ""))
        if not parsed.parse_success:
            category = "parse_failure"
        elif not parsed.legal:
            category = "illegal"
        else:
            category = "legal_parsed"
        return {
            "predicted_move_uci": parsed.parsed_uci,
            "parse_success": parsed.parse_success,
            "is_legal": parsed.legal,
            "runtime_seconds": runtime,
            "diagnostics": {
                "outcome_category": category,
                "raw_model_response": parsed.raw_response,
                "strict_status": parsed.status,
                "strict_error": parsed.error,
                "strict_san": parsed.san,
            },
        }


class MockClassicAdapter(ClassicModelAdapter):
    """Fixture-only adapter that emits predefined predictions."""

    def __init__(self, predictions: dict[str, str], model_id: str = "MOCK_CLASSIC"):
        super().__init__(model_id, "fixture", "test_only")
        self.predictions = dict(predictions)

    def predict(self, sample) -> dict[str, Any]:
        """Return a deterministic fixture prediction for one sample."""

        predicted = self.predictions.get(sample.heldout_id)
        legal = None
        if predicted is not None:
            try:
                legal = chess.Move.from_uci(predicted) in chess.Board(sample.fen).legal_moves
            except ValueError:
                legal = False
        return {"predicted_move_uci": predicted, "is_legal": legal, "ranked_moves_uci": [predicted] if predicted else []}
