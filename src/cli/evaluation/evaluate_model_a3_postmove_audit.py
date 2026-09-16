"""CLI for A3 post-move and opponent-response diagnostic audit."""

from __future__ import annotations

import argparse
import sys

import torch

from src.evaluation.model_a.model_a3_postmove_audit import (
    A3PostMoveAuditConfig,
    run_postmove_audit,
)


def parse_args():
    """Parse A3 post-move audit options."""

    parser = argparse.ArgumentParser(
        description="Run post-hoc A3 post-move/opponent-response audit."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--non-blocking", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--output-dir", default="artifacts/model_a3_postmove_audit")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/model_a3_legal_move_scorer_no_timing/best.pt",
    )
    parser.add_argument("--pyg-root", default="data/pyg")
    return parser.parse_args()


def main():
    """Run the diagnostic audit and print the compact summary."""

    args = parse_args()
    config = A3PostMoveAuditConfig(
        batch_size=args.batch_size,
        device=args.device,
        non_blocking=args.non_blocking,
        amp=args.amp,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        pyg_root=args.pyg_root,
    )
    try:
        summary = run_postmove_audit(config)
    except (FileNotFoundError, RuntimeError) as exc:
        print("MODEL_A3_POSTMOVE_AUDIT_BLOCKED")
        print(str(exc))
        return 1

    print("MODEL_A3_POSTMOVE_AUDIT_COMPLETE")
    print(f"A3_PARITY={summary['a3_parity']['status']}")
    print(f"TEST_N={summary['test_n']}")
    print(f"A3_TOP1={summary['a3_top1']}")
    print(f"A3_TOP5={summary['a3_top5']}")
    print(f"TOP5_RERANKER_CEILING={summary['top5_reranker_ceiling']}")
    print(f"RECOVERABLE_TOP5_ERRORS={summary['recoverable_top5_errors']}")
    print(f"UNRECOVERABLE_TOP5_ERRORS={summary['unrecoverable_top5_errors']}")
    print(
        "LEVEL1_CANDIDATES_PER_SECOND="
        f"{summary['runtime']['level1']['candidates_per_second']}"
    )
    print(
        "LEVEL2_CANDIDATES_PER_SECOND="
        f"{summary['runtime']['level2']['candidates_per_second']}"
    )
    print(
        "LEVEL2_RESPONSES_PER_SECOND="
        f"{summary['runtime']['level2']['responses_per_second']}"
    )
    print(f"A4_ARCHITECTURE_EVIDENCE={summary['a4_architecture_evidence']}")
    print("A4_IMPLEMENTED=NO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
