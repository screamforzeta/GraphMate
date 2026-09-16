"""CLI for the frozen A3 prediction headroom and error audit."""

from __future__ import annotations

import argparse
import sys

import torch

from src.evaluation.model_a.model_a3_error_audit import (
    A3ErrorAuditConfig,
    discover_stockfish,
    refresh_report_from_rows,
    run_audit,
)


def parse_args():
    """Parse A3 error audit options."""

    parser = argparse.ArgumentParser(
        description="Run read-only A3 prediction headroom and error audit."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--non-blocking", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--output-dir", default="artifacts/model_a3_error_audit")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/model_a3_legal_move_scorer_no_timing/best.pt",
    )
    parser.add_argument("--pyg-root", default="data/pyg")
    parser.add_argument("--refresh-report-only", action="store_true")
    parser.add_argument("--stockfish-discovery-only", action="store_true")
    parser.add_argument("--stockfish", action="store_true")
    parser.add_argument("--stockfish-path", default=None)
    parser.add_argument("--stockfish-depth", type=int, default=18)
    parser.add_argument("--stockfish-max-samples", type=int, default=None)
    parser.add_argument("--stockfish-threads", type=int, default=1)
    parser.add_argument("--stockfish-hash-mb", type=int, default=128)
    return parser.parse_args()


def build_config(args):
    """Build runtime config from parsed CLI args."""

    return A3ErrorAuditConfig(
        batch_size=args.batch_size,
        device=args.device,
        non_blocking=args.non_blocking,
        amp=args.amp,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        pyg_root=args.pyg_root,
        stockfish=args.stockfish,
        stockfish_path=args.stockfish_path,
        stockfish_depth=args.stockfish_depth,
        stockfish_max_samples=args.stockfish_max_samples,
        stockfish_threads=args.stockfish_threads,
        stockfish_hash_mb=args.stockfish_hash_mb,
    )


def main():
    """Run the requested audit mode."""

    args = parse_args()
    if args.stockfish_discovery_only:
        discovery = discover_stockfish(args.stockfish_path)
        print("STOCKFISH_DISCOVERY")
        print(f"available={discovery['available']}")
        print(f"path={discovery.get('path')}")
        print(f"version={discovery.get('version')}")
        if discovery.get("error"):
            print(f"error={discovery['error']}")
        return 0

    config = build_config(args)
    try:
        if args.refresh_report_only:
            summaries, paths = refresh_report_from_rows(config)
        else:
            summaries, paths = run_audit(config)
    except FileNotFoundError as exc:
        print("MODEL_A3_ERROR_AUDIT_BLOCKED")
        print(str(exc))
        return 1

    validation = summaries["validation"]
    test = summaries["test"]
    print("MODEL_A3_ERROR_AUDIT_COMPLETE")
    print(f"validation_summary={paths['validation_summary']}")
    print(f"test_summary={paths['test_summary']}")
    print(f"report={paths['report']}")
    print(f"validation_n={validation['actual_n']}")
    print(f"test_n={test['actual_n']}")
    print(f"validation_top1={validation['topk']['top1']}")
    print(f"validation_top3={validation['topk']['top3']}")
    print(f"validation_top5={validation['topk']['top5']}")
    print(f"validation_top10={validation['topk']['top10']}")
    print(f"test_top1={test['topk']['top1']}")
    print(f"test_top3={test['topk']['top3']}")
    print(f"test_top5={test['topk']['top5']}")
    print(f"test_top10={test['topk']['top10']}")
    print(f"A3_TEST_PARITY={test['parity']['status']}")
    print(f"STOCKFISH_ANALYSIS_AVAILABLE={test['stockfish']['available']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
