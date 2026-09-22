"""Markdown report assembly for final experiment analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def fmt(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def build_report(tables: dict[str, Any], figures: list[str], out_dir: Path) -> str:
    rq2 = tables["rq2_results"]
    rq1_rows = tables["rq1_results"]["rows"]
    tests = tables["statistical_tests"]
    lines = [
        "# Final Experiment Summary",
        "",
        "This report is generated from existing official artifacts only. No training, official inference, LLM inference, Popeye verification, dataset rebuild, or checkpoint mutation is performed.",
        "",
        "## Experiment Inventory",
        f"Included experiments: {len(tables['experiment_inventory']['included'])}. Excluded/diagnostic artifacts: {len(tables['experiment_inventory']['excluded'])}.",
        "",
        "Excluded artifacts include the invalid first Qwen 4B classic HTTP-404 attempt, smoke tests, and GPT-OSS protocol-failure evidence outside the primary comparison.",
        "",
        "## Dataset and Evaluation Protocols",
        "Lichess is the only benchmark used for the literal timed-GNN-vs-LLM comparison because Model B timing inputs are available there. YACPDB classic is an external generalization benchmark for A3/A4 and LLMs; Model B is N/A - TIMING_UNAVAILABLE.",
        "",
        "## Lichess Primary Results",
        md_table(tables["lichess_primary"], ["model", "N", "Top1_percent", "Top3_percent", "Top5_percent", "mean_rank", "protocol_notes"]),
        "",
        "## Timing Ablation / RQ2",
        md_table(tables["timing_ablation"], ["model", "timing_mode", "N", "Top1_percent", "Top3_percent", "Top5_percent", "delta_Top1_vs_A3_pp"]),
        "",
        f"RQ2 bounded conclusion: {rq2['bounded_conclusion']}",
        "",
        "## LLM Comparison / RQ1",
        md_table(rq1_rows, ["mate_depth", "N", "B_Top1_percent", "Qwen4_Top1_percent", "Qwen9_Top1_percent", "B_minus_Qwen4_pp", "B_minus_Qwen9_pp"]),
        "",
        "RQ1 is computed on Lichess only. A3 and A4 are contextual no-timing graph baselines in the associated depth table and figures.",
        "",
        "## External Classic Benchmark",
        md_table(tables["classic_primary"], ["model", "ALL200_Top1_percent", "Top3_percent", "Top5_percent", "correct_count", "Popeye193_Top1_percent", "notes"]),
        "",
        "Classic mate-depth buckets contain only N=20 problems each, so interval estimates are wide and individual buckets should not be overinterpreted.",
        "",
        "## LLM Failure Analysis",
        md_table(tables["llm_failure_analysis"], ["benchmark", "model", "outcome", "count", "percent"]),
        "",
        "The LLM analysis separates chess correctness, legality, strict parse compliance, and runtime failure.",
        "",
        "## External Generalization",
        md_table(tables["generalization"], ["model", "lichess_Top1_percent", "classic_Top1_percent", "classic_minus_lichess_pp"]),
        "",
        "Performance under the external composition distribution was substantially lower for the graph models; this is an out-of-distribution/external-generalization result, not an IID claim that the benchmark is simply harder.",
        "",
        "## Statistical Analysis",
        f"Lichess A3 vs B paired test: {tests['paired_tests']['Lichess_A3_vs_B']}",
        "",
        f"Classic A3 vs A4 paired test: {tests['paired_tests']['Classic_A3_vs_A4']}",
        "",
        f"A3 vs B Top1 effect size: {tests['effect_sizes']['A3_vs_B_Top1_delta_pp']:.6f} percentage points.",
        "",
        "Wilson 95% confidence intervals are stored in `data/statistical_tests.json` and included in primary/depth CSV outputs.",
        "",
        "## Limitations",
        "Timing is synthetic and rating-conditioned. The classic benchmark has no timing fields and cannot answer the timed-model comparison. GPT-OSS is retained as protocol-failure evidence, not as a primary benchmark row. Relaxed LLM parser results remain diagnostic and are not substituted for strict results.",
        "",
        "## Direct Answers",
        "RQ1: On Lichess MateIn1 through MateIn5, Model B Top1 is far above both strict Qwen baselines for every depth with available artifacts.",
        "",
        "RQ2: The implemented synthetic timing signal did not improve predictive performance relative to A3; Top1 changed by a negative amount in the controlled comparison.",
        "",
        "## Missing Analyses / Unavailable Data",
        "Model B on YACPDB classic is unavailable because timing inputs are unavailable. No unsupported YACPDB timing values are generated.",
        "",
        "## Figures",
    ]
    lines.extend(f"- `{Path(path).relative_to(out_dir)}`" for path in figures)
    lines.extend([
        "",
        "## Artifact Provenance",
        "See `source_manifest.json` and `data/experiment_inventory.json`.",
        "",
        "## Validation",
    ])
    lines.extend(f"- {item}" for item in tables["validation"])
    lines.append("")
    return "\n".join(lines)

