"""Build final experiment analysis outputs from frozen artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.final_experiments.artifacts import discover
from src.analysis.final_experiments.figures import build_figures
from src.analysis.final_experiments.report import build_report
from src.analysis.final_experiments.tables import build_tables, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/final_analysis"))
    args = parser.parse_args()

    discovered = discover(args.root)
    tables = build_tables(discovered)
    out_dir = args.root / args.out_dir
    write_outputs(out_dir, tables)
    figures = build_figures(tables, out_dir)
    report = build_report(tables, figures, out_dir)
    (out_dir / "final_experiment_summary.md").write_text(report, encoding="utf-8")
    print(f"Final analysis written to {out_dir}")


if __name__ == "__main__":
    main()

