"""CLI for frozen Model A3 vs frozen Model A4 terminal evaluation."""

from __future__ import annotations

import argparse
import sys

import torch

from src.evaluation.model_a.model_a3_vs_a4_terminal import (
    A3A4TerminalConfig,
    run_terminal_evaluation,
    terminal_preflight,
    write_terminal_outputs,
)


def parse_args():
    """Parse A3-vs-A4 terminal evaluator arguments."""

    parser = argparse.ArgumentParser(
        description="Terminal test evaluation for frozen A3 vs frozen A4."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--non-blocking", action="store_true", default=True)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--run-terminal-test", action="store_true")
    parser.add_argument("--a3-checkpoint", default="artifacts/model_a3_legal_move_scorer_no_timing/best.pt")
    parser.add_argument("--a4-checkpoint", default="artifacts/model_a4_postmove_gnn_reranker/best.pt")
    parser.add_argument("--a4-training-summary", default="artifacts/model_a4_postmove_gnn_reranker/training_summary.json")
    parser.add_argument("--test-csv", default="data/final/puzzles/test.csv")
    parser.add_argument("--pyg-root", default="data/pyg")
    parser.add_argument("--output-dir", default="artifacts/model_a4_terminal_evaluation")
    return parser.parse_args()


def build_config(args):
    """Build evaluator config from CLI args."""

    return A3A4TerminalConfig(
        batch_size=args.batch_size,
        device=args.device,
        non_blocking=args.non_blocking,
        amp=args.amp,
        a3_checkpoint=args.a3_checkpoint,
        a4_checkpoint=args.a4_checkpoint,
        a4_training_summary=args.a4_training_summary,
        pyg_root=args.pyg_root,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
    )


def print_preflight(preflight):
    """Print preflight status in a stable terminal format."""

    print("MODEL_A3_VS_A4_TERMINAL_PREFLIGHT")
    for key, value in preflight.items():
        print(f"{key} = {value}")


def main():
    """Run preflight or explicit terminal evaluation."""

    args = parse_args()
    config = build_config(args)
    preflight = terminal_preflight(config)
    print_preflight(preflight)
    if not args.run_terminal_test:
        print("TERMINAL_TEST_EXECUTED = NO")
        print("Use --run-terminal-test to perform the one-time frozen test evaluation.")
        return 0
    if not preflight["evaluator_ready"]:
        print("TERMINAL_TEST_BLOCKED = missing frozen checkpoint(s)")
        return 1
    summary, rank_rows = run_terminal_evaluation(config)
    paths = write_terminal_outputs(summary, rank_rows, config.output_dir)
    print("MODEL_A3_VS_A4_TERMINAL_EVALUATION_COMPLETE")
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"SHARED_TEST_POPULATION = {summary['shared_test_population']}")
    print(f"A3_TEST_PARITY = {summary['a3']['parity']['status']}")
    print(f"A4_END_TO_END_TOP1 = {summary['a4']['end_to_end_top1']}")
    print(f"DELTA_TOP1_PP = {summary['paired_transitions']['delta_top1_pp']}")
    if not summary["shared_test_population"] or summary["a3"]["parity"]["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
