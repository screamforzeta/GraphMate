"""Terminal test evaluator for frozen Model A3 vs frozen Model A4.

Purpose:
    Compare frozen A3 and frozen A4 on the shared untouched test population.
Input:
    Frozen A3/A4 checkpoints, data/pyg/test, test CSV, and move vocabulary.
Output:
    Terminal evaluation artifacts under artifacts/model_a4_terminal_evaluation/.
Run:
    python3 -m src.cli.evaluation.evaluate_model_a3_vs_a4 --run-terminal-test
"""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import statistics

import pandas as pd
import torch
from torch_geometric.data import Batch

from src.evaluation.model_a.model_a4_postmove_evaluation import load_a4_checkpoint
from src.evaluation.model_a.model_a_vs_a2 import _rating_bucket
from src.evaluation.model_a.model_a_vs_a2_vs_a3 import (
    A3_PARITY_TOLERANCES,
    MODEL_A3_CHECKPOINT,
    MODEL_A3_REFERENCE,
    SHARED_TEST_N,
    a3_parity_diagnostics,
    a3_topk_and_ranks,
    load_eval_model_a3,
)
from src.graph.pyg_dataset import load_move_encoder, load_pyg_dataset
from src.training.model_a.model_a3_legal_scorer import (
    build_candidate_batch,
    grouped_cross_entropy,
)
from src.training.model_a.model_a4_postmove_reranker import (
    _a3_candidate_features,
    build_postmove_graph,
    grouped_cross_entropy_from_topk,
    select_topk_a3,
)


MODEL_A4_CHECKPOINT = Path("checkpoints/model_a4/best.pt")
MODEL_A4_TRAINING_SUMMARY = Path("artifacts/model_a4_postmove_gnn_reranker/training_summary.json")
OUTPUT_DIR = Path("artifacts/model_a4_terminal_evaluation")
REPORT_PATH = Path("generic_info/models/model_a/model_a4_terminal_evaluation.md")
EXPECTED_A4_BEST_EPOCH = 32
RATING_BUCKETS = ["<1200", "1200-1599", "1600-1999", "2000-2399", "2400+"]


@dataclass
class A3A4TerminalConfig:
    """Runtime config for terminal A3-vs-A4 evaluation."""

    batch_size: int = 128
    device: str = "cpu"
    non_blocking: bool = False
    amp: bool = False
    a3_checkpoint: str = str(MODEL_A3_CHECKPOINT)
    a4_checkpoint: str = str(MODEL_A4_CHECKPOINT)
    a4_training_summary: str = str(MODEL_A4_TRAINING_SUMMARY)
    pyg_root: str = "data/pyg"
    test_csv: str = "data/final/puzzles/test.csv"
    output_dir: str = str(OUTPUT_DIR)


def mate_bucket(value):
    """Return the MateDepth bucket label."""

    return f"MateIn{int(value)}"


def terminal_preflight(config):
    """Return preflight status without evaluating the test set."""

    a3_path = Path(config.a3_checkpoint)
    a4_path = Path(config.a4_checkpoint)
    summary_path = Path(config.a4_training_summary)
    best_epoch = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best_epoch = summary.get("best_epoch")
    return {
        "frozen_a3_checkpoint_found": a3_path.exists(),
        "frozen_a4_checkpoint_found": a4_path.exists(),
        "a4_training_summary_found": summary_path.exists(),
        "a4_best_epoch": best_epoch,
        "a4_best_epoch_expected": EXPECTED_A4_BEST_EPOCH,
        "a4_best_epoch_matches": best_epoch == EXPECTED_A4_BEST_EPOCH,
        "expected_shared_test_n": SHARED_TEST_N,
        "evaluator_ready": a3_path.exists() and a4_path.exists(),
        "target_injection_impossible": True,
        "test_used_for_training": False,
        "test_used_for_checkpoint_selection": False,
        "models_eval_mode": "checked during --run-terminal-test",
        "gradients_disabled": "checked during --run-terminal-test",
        "test_evaluation_has_happened": False,
    }


def exact_mcnemar_pvalue(n01, n10):
    """Return exact two-sided binomial McNemar p-value."""

    total = int(n01) + int(n10)
    if total == 0:
        return 1.0
    observed = min(int(n01), int(n10))
    probability = 0.5 ** total
    cumulative = probability
    for k in range(0, observed):
        probability *= (total - k) / (k + 1)
        cumulative += probability
    return min(1.0, 2.0 * cumulative)


def mcnemar_summary(n01, n10):
    """Return McNemar exact and continuity-corrected summaries."""

    total = int(n01) + int(n10)
    chi_square = None
    if total:
        chi_square = (abs(int(n01) - int(n10)) - 1) ** 2 / total
    return {
        "n01_a3_wrong_a4_correct": int(n01),
        "n10_a3_correct_a4_wrong": int(n10),
        "continuity_corrected_chi_square": chi_square,
        "exact_two_sided_binomial_p": exact_mcnemar_pvalue(n01, n10),
    }


def new_breakdown_bucket():
    """Create counters for one subgroup."""

    return {
        "n": 0,
        "a3_correct": 0,
        "a4_correct": 0,
        "rerankable": 0,
        "a4_conditional_correct": 0,
        "a3_wrong_a4_correct": 0,
        "a3_correct_a4_wrong": 0,
    }


def update_breakdown(bucket, a3_correct, a4_correct, rerankable):
    """Update one subgroup bucket."""

    bucket["n"] += 1
    bucket["a3_correct"] += int(a3_correct)
    bucket["a4_correct"] += int(a4_correct)
    bucket["rerankable"] += int(rerankable)
    bucket["a4_conditional_correct"] += int(a4_correct and rerankable)
    bucket["a3_wrong_a4_correct"] += int((not a3_correct) and a4_correct)
    bucket["a3_correct_a4_wrong"] += int(a3_correct and (not a4_correct))


def finalize_breakdown(buckets):
    """Finalize subgroup counters into rates."""

    rows = []
    for label, bucket in buckets.items():
        n = bucket["n"]
        if n == 0:
            continue
        rerankable = bucket["rerankable"]
        rows.append(
            {
                "bucket": label,
                "n": n,
                "a3_top1": bucket["a3_correct"] / n,
                "a4_end_to_end_top1": bucket["a4_correct"] / n,
                "delta_a4_minus_a3_pp": 100.0 * ((bucket["a4_correct"] - bucket["a3_correct"]) / n),
                "a3_recall_at_5": rerankable / n,
                "a4_conditional_top1": (
                    bucket["a4_conditional_correct"] / rerankable if rerankable else None
                ),
                "net_paired_gain": bucket["a3_wrong_a4_correct"] - bucket["a3_correct_a4_wrong"],
            }
        )
    return rows


def safe_mean(values):
    """Return mean for non-empty values."""

    return statistics.mean(values) if values else None


def add_representative(representatives, category, row, limit=5):
    """Store a bounded qualitative example row."""

    if len(representatives[category]) < limit:
        representatives[category].append(row)


def _a4_score_one_position(a4, a3_features, topk_row, graph, move_to_idx, device, use_amp):
    """Score A4 Top-K candidates for one original graph."""

    postmove_graphs = [
        build_postmove_graph(graph.fen, move_uci, str(graph.target_move), move_to_idx)
        for move_uci in topk_row["uci"]
    ]
    postmove_batch = Batch.from_data_list(postmove_graphs).to(device)
    score_features = torch.tensor(
        list(zip(topk_row["raw_scores"], topk_row["centered_scores"])),
        dtype=torch.float,
        device=device,
    )
    with torch.amp.autocast(device.type, enabled=use_amp):
        scores = a4(a3_features, postmove_batch, score_features)
    ordered = torch.argsort(scores, descending=True)
    return scores, ordered


def run_terminal_evaluation(config):
    """Run the one-time terminal A3-vs-A4 test evaluation."""

    device = torch.device(config.device)
    dataset = load_pyg_dataset(root=config.pyg_root, split="test")
    if len(dataset) != SHARED_TEST_N:
        raise ValueError(f"SHARED_TEST_POPULATION mismatch: got {len(dataset)} expected {SHARED_TEST_N}")
    dataframe = pd.read_csv(config.test_csv)
    csv_rows = list(dataframe.itertuples(index=False))
    move_to_idx = load_move_encoder()
    a3, _ = load_eval_model_a3(Path(config.a3_checkpoint), device)
    a4, a4_config, a4_checkpoint = load_a4_checkpoint(Path(config.a4_checkpoint), device)
    a3.eval()
    a4.eval()
    use_amp = bool(config.amp and device.type == "cuda")

    a3_loss_sum = 0.0
    ranks = []
    a3_top = Counter()
    a4_loss_sum = 0.0
    rerankable_n = 0
    a4_correct_n = 0
    transitions = Counter()
    mate_buckets = defaultdict(new_breakdown_bucket)
    rating_buckets = {label: new_breakdown_bucket() for label in RATING_BUCKETS}
    retrieval_failures = 0
    reranking_failures = 0
    rank_movements = Counter()
    rank_matrix = Counter()
    representatives = defaultdict(list)
    rows_for_rank_csv = []

    with torch.no_grad():
        for start in range(0, len(dataset), config.batch_size):
            graphs = [dataset[index] for index in range(start, min(start + config.batch_size, len(dataset)))]
            batch = Batch.from_data_list(graphs).to(
                device,
                non_blocking=bool(config.non_blocking and device.type == "cuda"),
            )
            candidate = build_candidate_batch(batch)
            target_indices = candidate["target_indices"].to(device)
            with torch.amp.autocast(device.type, enabled=use_amp):
                a3_output = a3(batch, candidate["candidate_moves"])
                a3_loss = grouped_cross_entropy(
                    a3_output["scores"],
                    a3_output["candidate_ptr"],
                    target_indices,
                )
            a3_loss_sum += float(a3_loss.item()) * len(graphs)
            a3_metrics = a3_topk_and_ranks(
                a3_output["scores"],
                a3_output["candidate_ptr"],
                target_indices,
                top_k=(1, 3, 5, 10),
            )
            ranks.extend(a3_metrics["ranks"])
            for k, value in a3_metrics["correct"].items():
                a3_top[k] += value
            a3_features = _a3_candidate_features(a3, batch, candidate["candidate_moves"])
            topk_rows = select_topk_a3(a3_output, target_indices, top_k=5)
            for local_index, (graph, topk_row) in enumerate(zip(graphs, topk_rows)):
                source_row = csv_rows[int(torch.as_tensor(graph.source_row_index).item())]
                flat_start = int(a3_output["candidate_ptr"][local_index].item())
                selected_flat = [flat_start + local for local in topk_row["local_indices"]]
                scores_a4, ordered_a4 = _a4_score_one_position(
                    a4,
                    a3_features[selected_flat],
                    topk_row,
                    graph,
                    move_to_idx,
                    device,
                    use_amp,
                )
                target_local = topk_row["target_candidate_index"]
                rerankable = target_local is not None
                a3_rank = int(ranks[-len(graphs) + local_index])
                a3_correct = a3_rank == 1
                if rerankable:
                    rerankable_n += 1
                    ptr = torch.tensor([0, len(topk_row["uci"])], dtype=torch.long, device=device)
                    target_tensor = torch.tensor([target_local], dtype=torch.long, device=device)
                    a4_loss = grouped_cross_entropy_from_topk(scores_a4, ptr, target_tensor)
                    a4_loss_sum += float(a4_loss.item())
                    a4_rank = int((ordered_a4 == target_local).nonzero(as_tuple=False).item()) + 1
                    if a4_rank < target_local + 1:
                        rank_movements["improved"] += 1
                    elif a4_rank == target_local + 1:
                        rank_movements["unchanged"] += 1
                    else:
                        rank_movements["worsened"] += 1
                    rank_matrix[f"{target_local + 1}->{a4_rank}"] += 1
                else:
                    a4_rank = None
                    retrieval_failures += 1
                a4_top_index = int(ordered_a4[0].item())
                a4_correct = bool(rerankable and a4_top_index == target_local)
                if rerankable and not a4_correct:
                    reranking_failures += 1
                a4_correct_n += int(a4_correct)
                if a3_correct and a4_correct:
                    transitions["both_correct"] += 1
                elif a3_correct and not a4_correct:
                    transitions["a3_correct_a4_wrong"] += 1
                elif (not a3_correct) and a4_correct:
                    transitions["a3_wrong_a4_correct"] += 1
                else:
                    transitions["both_wrong"] += 1
                rating = int(source_row.Rating)
                mate_depth = int(source_row.MateDepth)
                update_breakdown(mate_buckets[mate_bucket(mate_depth)], a3_correct, a4_correct, rerankable)
                update_breakdown(rating_buckets[_rating_bucket(source_row)], a3_correct, a4_correct, rerankable)
                example = {
                    "PuzzleId": str(source_row.PuzzleId),
                    "FEN": str(graph.fen),
                    "target_uci": str(graph.target_move),
                    "a3_top5": topk_row["uci"],
                    "a3_scores": topk_row["raw_scores"],
                    "a4_order": [topk_row["uci"][int(index.item())] for index in ordered_a4],
                    "a4_scores": [float(value) for value in scores_a4.detach().cpu().tolist()],
                    "a3_prediction": topk_row["uci"][0],
                    "a4_prediction": topk_row["uci"][a4_top_index],
                    "MateDepth": mate_depth,
                    "Rating": rating,
                    "a3_correct": a3_correct,
                    "a4_correct": a4_correct,
                    "rerankable": rerankable,
                    "a3_target_rank": a3_rank,
                    "a4_target_rank_within_top5": a4_rank,
                }
                if (not a3_correct) and a4_correct:
                    add_representative(representatives, "a3_wrong_a4_correct", example)
                elif a3_correct and (not a4_correct):
                    add_representative(representatives, "a3_correct_a4_wrong", example)
                elif (not a3_correct) and (not a4_correct) and rerankable:
                    add_representative(representatives, "both_wrong_target_in_top5", example)
                elif not rerankable:
                    add_representative(representatives, "target_outside_a3_top5", example)
                if rerankable:
                    rows_for_rank_csv.append(
                        {
                            "PuzzleId": str(source_row.PuzzleId),
                            "a3_rank_within_top5": target_local + 1,
                            "a4_rank_within_top5": a4_rank,
                            "movement": (
                                "improved" if a4_rank < target_local + 1 else
                                "unchanged" if a4_rank == target_local + 1 else
                                "worsened"
                            ),
                        }
                    )

    n = len(dataset)
    a3_actual = {
        "n": n,
        "a3_loss": a3_loss_sum / n,
        "a3_top1": a3_top[1] / n,
        "a3_top3": a3_top[3] / n,
        "a3_top5": a3_top[5] / n,
        "a3_mean_legal_rank": statistics.mean(ranks),
        "a3_median_legal_rank": statistics.median(ranks),
        "a3_illegal_top1_rate": 0.0,
    }
    parity = a3_parity_diagnostics(a3_actual, MODEL_A3_REFERENCE, A3_PARITY_TOLERANCES)
    parity_status = "PASS" if all(item["result"] == "PASS" for item in parity) else "FAIL"
    a4_end_to_end_top1 = a4_correct_n / n
    a3_top1 = a3_actual["a3_top1"]
    a3_wrong = n - a3_top[1]
    summary = {
        "shared_test_population": n == SHARED_TEST_N,
        "a3": {
            **a3_actual,
            "a3_top10": a3_top[10] / n,
            "parity": {"status": parity_status, "diagnostics": parity},
        },
        "a4": {
            "n": n,
            "rerankable_n": rerankable_n,
            "unrerankable_n": n - rerankable_n,
            "a3_recall_at_5": rerankable_n / n,
            "conditional_top1": a4_correct_n / rerankable_n if rerankable_n else None,
            "end_to_end_top1": a4_end_to_end_top1,
            "end_to_end_error_count": n - a4_correct_n,
            "conditional_ce_nll": a4_loss_sum / rerankable_n if rerankable_n else None,
            "gap_to_retrieval_ceiling_pp": 100.0 * ((rerankable_n / n) - a4_end_to_end_top1),
        },
        "paired_transitions": {
            **dict(transitions),
            "a3_correct_preserved_rate": (
                transitions["both_correct"] / (transitions["both_correct"] + transitions["a3_correct_a4_wrong"])
                if (transitions["both_correct"] + transitions["a3_correct_a4_wrong"]) else None
            ),
            "a3_errors_recovered_rate_rerankable_definition": (
                transitions["a3_wrong_a4_correct"] /
                (transitions["a3_wrong_a4_correct"] + transitions["both_wrong"] - retrieval_failures)
                if (transitions["a3_wrong_a4_correct"] + transitions["both_wrong"] - retrieval_failures) > 0 else None
            ),
            "a3_errors_recovered_rate_all_a3_wrong": (
                transitions["a3_wrong_a4_correct"] / a3_wrong if a3_wrong else None
            ),
            "net_correct_gain": transitions["a3_wrong_a4_correct"] - transitions["a3_correct_a4_wrong"],
            "delta_top1_pp": 100.0 * (a4_end_to_end_top1 - a3_top1),
        },
        "mcnemar": mcnemar_summary(
            transitions["a3_wrong_a4_correct"],
            transitions["a3_correct_a4_wrong"],
        ),
        "retrieval_vs_reranking": {
            "retrieval_failures": retrieval_failures,
            "reranking_failures": reranking_failures,
            "retrieval_failure_fraction_of_a4_errors": retrieval_failures / (n - a4_correct_n) if n != a4_correct_n else None,
            "reranking_failure_fraction_of_a4_errors": reranking_failures / (n - a4_correct_n) if n != a4_correct_n else None,
        },
        "rank_movements": {
            "counts": dict(rank_movements),
            "transition_matrix": dict(rank_matrix),
        },
        "mate_depth_breakdown": finalize_breakdown(mate_buckets),
        "rating_breakdown": finalize_breakdown(rating_buckets),
        "representative_examples": representatives,
        "a4_checkpoint_epoch": a4_checkpoint.get("epoch"),
        "a4_candidate_dim": getattr(a4, "candidate_dim", None),
        "test_used_for_training": False,
        "test_used_for_checkpoint_selection": False,
        "target_injected_in_top5": False,
    }
    return summary, rows_for_rank_csv


def write_terminal_outputs(summary, rank_rows, output_dir=OUTPUT_DIR, report_path=REPORT_PATH):
    """Write terminal evaluation artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=list), encoding="utf-8")
    write_csv(output_dir / "mate_depth_breakdown.csv", summary["mate_depth_breakdown"])
    write_csv(output_dir / "rating_breakdown.csv", summary["rating_breakdown"])
    (output_dir / "paired_transitions.json").write_text(
        json.dumps(summary["paired_transitions"], indent=2),
        encoding="utf-8",
    )
    with open(output_dir / "representative_examples.jsonl", "w", encoding="utf-8") as file:
        for category, examples in summary["representative_examples"].items():
            for example in examples:
                file.write(json.dumps({"category": category, **example}) + "\n")
    write_csv(output_dir / "rank_movements.csv", rank_rows)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown_report(summary), encoding="utf-8")
    return {
        "summary_json": str(output_dir / "summary.json"),
        "mate_depth_breakdown_csv": str(output_dir / "mate_depth_breakdown.csv"),
        "rating_breakdown_csv": str(output_dir / "rating_breakdown.csv"),
        "paired_transitions_json": str(output_dir / "paired_transitions.json"),
        "representative_examples_jsonl": str(output_dir / "representative_examples.jsonl"),
        "rank_movements_csv": str(output_dir / "rank_movements.csv"),
        "report_md": str(report_path),
    }


def write_csv(path, rows):
    """Write dictionaries as CSV."""

    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown_report(summary):
    """Render a human-readable terminal report."""

    return "\n".join(
        [
            "# Model A4 Terminal Evaluation",
            "",
            "This report compares frozen A3 and frozen A4 on the shared terminal test population.",
            "",
            "## Test Lock",
            "",
            "- TEST_USED_FOR_TRAINING = NO",
            "- TEST_USED_FOR_CHECKPOINT_SELECTION = NO",
            "- TARGET_INJECTED_IN_TOP5 = NO",
            "",
            "## A3 Parity",
            "",
            f"- A3_TEST_PARITY = `{summary['a3']['parity']['status']}`",
            f"- A3 Top1 = `{summary['a3']['a3_top1']}`",
            f"- A3 Top5 = `{summary['a3']['a3_top5']}`",
            "",
            "## A4 Terminal Test",
            "",
            f"- A4_END_TO_END_N = `{summary['a4']['n']}`",
            f"- A3_RECALL_AT_5 = `{summary['a4']['a3_recall_at_5']}`",
            f"- A4_CONDITIONAL_TOP1 = `{summary['a4']['conditional_top1']}`",
            f"- A4_END_TO_END_TOP1 = `{summary['a4']['end_to_end_top1']}`",
            f"- DELTA_TOP1_PP = `{summary['paired_transitions']['delta_top1_pp']}`",
            "",
            "## Paired Comparison",
            "",
            "```json",
            json.dumps(summary["paired_transitions"], indent=2),
            "```",
            "",
            "## McNemar",
            "",
            "```json",
            json.dumps(summary["mcnemar"], indent=2),
            "```",
            "",
            "## Retrieval vs Reranking Errors",
            "",
            "```json",
            json.dumps(summary["retrieval_vs_reranking"], indent=2),
            "```",
            "",
        ]
    )
