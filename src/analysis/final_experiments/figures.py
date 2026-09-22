"""Matplotlib figures for final experiment analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, out_dir: Path, name: str, include_pdf: bool = False) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    formats = ["svg", "png"]
    if include_pdf:
        formats.insert(1, "pdf")
    for ext in formats:
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _bar(ax, labels, values, title, ylabel="Top1 (%)"):
    ax.bar(labels, values, color=["#2f6f73", "#d08c35", "#5b6fa8", "#9b4d64", "#607d3b", "#6b6b6b"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(100, max([v for v in values if v is not None] or [0]) * 1.15))
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)


def build_figures(tables: dict[str, Any], out_dir: Path, include_pdf: bool = False) -> list[str]:
    made: list[str] = []
    fig_dir = out_dir / "figures"
    if not include_pdf and fig_dir.exists():
        for stale_pdf in fig_dir.glob("*.pdf"):
            stale_pdf.unlink()

    graph_rows = [r for r in tables["lichess_primary"] if r["model"] in {"A", "A1", "A2", "A3", "B"}]
    labels = [r["model"] for r in graph_rows]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    for offset, metric, label in [(-width, "Top1_percent", "Top1"), (0, "Top3_percent", "Top3"), (width, "Top5_percent", "Top5")]:
        ax.bar([i + offset for i in x], [r[metric] for r in graph_rows], width=width, label=label)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Lichess Top-K Comparison")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    made += _save(fig, fig_dir, "lichess_topk_comparison", include_pdf=include_pdf)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in ["A3", "B", "A4", "Qwen 3.5 4B strict", "Qwen 3.5 9B strict"]:
        rows = [r for r in tables["lichess_by_mate_depth"] if r["model"] == model]
        ax.plot([r["mate_depth"].replace("MateIn", "M") for r in rows], [r["Top1_percent"] for r in rows], marker="o", label=model)
    ax.set_title("Lichess Top1 by Mate Depth")
    ax.set_ylabel("Top1 (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    made += _save(fig, fig_dir, "lichess_mate_depth_top1", include_pdf=include_pdf)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in ["A3", "A4"]:
        rows = [r for r in tables["lichess_by_rating"] if r["model"] == model]
        ax.plot([r["rating_bucket"] for r in rows], [r["Top1_percent"] for r in rows], marker="o", label=model)
    ax.set_title("Lichess Top1 by Rating Bucket")
    ax.set_ylabel("Top1 (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.25)
    ax.legend()
    made += _save(fig, fig_dir, "lichess_rating_top1", include_pdf=include_pdf)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = tables["timing_ablation"]
    _bar(ax, [r["model"].replace("_", "\n") for r in rows], [r["Top1_percent"] for r in rows], "Timing Ablation")
    made += _save(fig, fig_dir, "timing_ablation", include_pdf=include_pdf)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = [r for r in tables["classic_primary"] if r["model"] != "B"]
    _bar(ax, [r["model"].replace(" strict", "") for r in rows], [r["ALL200_Top1_percent"] for r in rows], "Classic Benchmark Top1")
    made += _save(fig, fig_dir, "classic_top1_comparison", include_pdf=include_pdf)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for model in ["A3", "A4", "Qwen 3.5 4B strict", "Qwen 3.5 9B strict"]:
        rows = [r for r in tables["classic_by_mate_depth"] if r["model"] == model]
        ax.plot([r["mate_depth"] for r in rows], [r["Top1_percent"] for r in rows], marker="o", label=model)
    ax.set_title("Classic Top1 by Mate Depth (N=20 each)")
    ax.set_ylabel("Top1 (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    made += _save(fig, fig_dir, "classic_mate_depth_top1", include_pdf=include_pdf)

    for benchmark, name in [("Lichess", "llm_failure_composition_lichess"), ("YACPDB classic", "llm_failure_composition_classic")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        rows = [r for r in tables["llm_failure_analysis"] if r["benchmark"] == benchmark]
        models = list(dict.fromkeys(r["model"] for r in rows))
        outcomes = ["correct", "wrong_legal", "illegal", "parse_failure", "runtime_failure"]
        bottoms = [0] * len(models)
        for outcome in outcomes:
            vals = [next(r["percent"] for r in rows if r["model"] == m and r["outcome"] == outcome) for m in models]
            ax.bar(models, vals, bottom=bottoms, label=outcome)
            bottoms = [a + b for a, b in zip(bottoms, vals)]
        ax.set_title(f"LLM Failure Composition: {benchmark}")
        ax.set_ylabel("Share (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=15)
        made += _save(fig, fig_dir, name, include_pdf=include_pdf)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = tables["generalization"]
    width = 0.35
    x = range(len(rows))
    ax.bar([i - width / 2 for i in x], [r["lichess_Top1_percent"] for r in rows], width=width, label="Lichess")
    ax.bar([i + width / 2 for i in x], [r["classic_Top1_percent"] for r in rows], width=width, label="YACPDB classic")
    ax.set_xticks(list(x), [r["model"].replace(" strict", "") for r in rows])
    ax.set_ylabel("Top1 (%)")
    ax.set_title("Lichess vs Classic External Generalization")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    made += _save(fig, fig_dir, "generalization_lichess_vs_classic", include_pdf=include_pdf)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["retrieval failures", "reranking failures", "A4 Top1 success"]
    values = [131, 38, 31]
    _bar(ax, labels, values, "A4 Classic Retrieval and Reranking", ylabel="Problems")
    ax.set_ylim(0, 200)
    made += _save(fig, fig_dir, "a4_classic_retrieval_reranking", include_pdf=include_pdf)

    return made
