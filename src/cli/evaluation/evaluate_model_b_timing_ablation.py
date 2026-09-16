"""CLI for official Model B timing ablation evaluation."""

from __future__ import annotations

import argparse

import torch

from src.evaluation.model_b.model_b_timing_ablation import (
    ModelBTimingAblationConfig,
    run_timing_ablation,
    write_outputs,
)


def parse_args():
    """Parse read-only Model B timing ablation options."""

    parser = argparse.ArgumentParser(
        description="Evaluate A3 vs Model B synthetic/neutral timing post-hoc."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--non-blocking", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--timing-dataset-root", default="data/pyg_puzzles_timing")
    parser.add_argument("--no-timing-dataset-root", default="data/pyg")
    parser.add_argument(
        "--model-b-checkpoint",
        default="artifacts/model_b_timing_legal_move_scorer/best.pt",
    )
    parser.add_argument(
        "--model-a3-checkpoint",
        default="artifacts/model_a3_legal_move_scorer_no_timing/best.pt",
    )
    parser.add_argument("--output-dir", default="artifacts/model_b_timing_ablation")
    parser.add_argument(
        "--skip-zero-timing-and-flag",
        action="store_true",
        help="Skip the secondary zero timing plus zero flag diagnostic.",
    )
    return parser.parse_args()


def main():
    """Run the read-only Model B timing ablation and write reports."""

    args = parse_args()
    config = ModelBTimingAblationConfig(
        batch_size=args.batch_size,
        device=args.device,
        non_blocking=args.non_blocking,
        amp=args.amp,
        timing_dataset_root=args.timing_dataset_root,
        no_timing_dataset_root=args.no_timing_dataset_root,
        model_b_checkpoint=args.model_b_checkpoint,
        model_a3_checkpoint=args.model_a3_checkpoint,
        output_dir=args.output_dir,
        include_zero_timing_and_flag=not args.skip_zero_timing_and_flag,
    )
    summary, paired_rows = run_timing_ablation(config)
    outputs = write_outputs(summary, paired_rows, args.output_dir)

    shared = summary["official_model_performance"]["shared_comparison_frame"]
    top1 = next(row for row in shared if row["metric"] == "Top1")
    top3 = next(row for row in shared if row["metric"] == "Top3")
    top5 = next(row for row in shared if row["metric"] == "Top5")
    neutral = summary["post_hoc_diagnostic_ablations"]["B_NEUTRAL_TIMING"]
    synthetic = summary["official_model_performance"]["B_SYNTHETIC_TIMING"]

    print("MODEL_B_TIMING_ABLATION_COMPLETE")
    print(f"summary_json={outputs['summary_json']}")
    print(f"report_md={outputs['report_md']}")
    print(f"paired_rows_json={outputs['paired_rows_json']}")
    print(f"MODEL_B_TERMINAL_PARITY={summary['model_b_terminal_parity']['status']}")
    print(f"A3_TOP1={top1['a3']}")
    print(f"B_SYNTHETIC_TOP1={top1['b']}")
    print(f"DELTA_B_MINUS_A3_TOP1_PP={top1['delta_b_minus_a3_pp']}")
    print(f"A3_TOP3={top3['a3']}")
    print(f"B_SYNTHETIC_TOP3={top3['b']}")
    print(f"DELTA_B_MINUS_A3_TOP3_PP={top3['delta_b_minus_a3_pp']}")
    print(f"A3_TOP5={top5['a3']}")
    print(f"B_SYNTHETIC_TOP5={top5['b']}")
    print(f"DELTA_B_MINUS_A3_TOP5_PP={top5['delta_b_minus_a3_pp']}")
    print(f"B_NEUTRAL_TOP1={neutral['top1']}")
    print(
        "DELTA_SYNTHETIC_MINUS_NEUTRAL_TOP1_PP="
        f"{(synthetic['top1'] - neutral['top1']) * 100}"
    )


if __name__ == "__main__":
    main()
