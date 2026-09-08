"""CLI for post-hoc Model A vs A2 vs A3 evaluation.

Purpose:
    Compare frozen Model A, A2, and A3 checkpoints on the shared PyG test set.
Input:
    Official best checkpoints, data/pyg/test, data/final/puzzles/test.csv,
    and artifacts/move_to_idx.json.
Output:
    artifacts/model_a_vs_a2_vs_a3_evaluation/summary.json and report.md.
Run:
    python3 -m src.evaluate_model_a_vs_a2_vs_a3 --device cuda --batch-size 128
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import torch

from src.evaluation.model_a_vs_a2_vs_a3 import (
    ComparisonA3Config,
    run_comparison,
    write_outputs,
)


def parse_args():
    """Parse A/A2/A3 evaluator CLI options."""

    parser = argparse.ArgumentParser(
        description="Run post-hoc Model A vs A2 vs A3 evaluation."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--non-blocking", action="store_true", default=True)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--test-csv", default="data/final/puzzles/test.csv")
    return parser.parse_args()


def main():
    """Run the evaluator and print parity summary."""

    args = parse_args()
    config = ComparisonA3Config(
        batch_size=args.batch_size,
        device=args.device,
        non_blocking=args.non_blocking,
        amp=args.amp,
    )
    dataframe = pd.read_csv(args.test_csv)
    try:
        summary = run_comparison(dataframe, config)
    except FileNotFoundError as error:
        print("MODEL_A_VS_A2_VS_A3_EVALUATION_BLOCKED")
        print(str(error))
        return 1
    paths = write_outputs(summary, config.output_dir)
    print("MODEL_A_VS_A2_VS_A3_EVALUATION_COMPLETE")
    print(f"summary_json: {paths['summary_json']}")
    print(f"report_md: {paths['report_md']}")
    print(f"MODEL_A_PARITY: {summary['model_a_parity']['status']}")
    print(f"MODEL_A2_PARITY: {summary['model_a2_parity']['status']}")
    print(f"MODEL_A3_PARITY: {summary['model_a3_parity']['status']}")
    print("A3 parity diagnostics:")
    for item in summary["model_a3_parity"].get("diagnostics", []):
        print(
            f"- {item['metric']}: expected={item['expected']} "
            f"actual={item['actual']} diff={item['abs_diff']} "
            f"tolerance={item['tolerance']} result={item['result']}"
        )
    if summary["model_a3_parity"]["status"] == "FAIL":
        print("A3 parity failed; subgroup interpretation is blocked.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
