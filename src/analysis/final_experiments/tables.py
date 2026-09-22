"""Build canonical final experiment tables from official artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .artifacts import DiscoveredArtifacts, read_json, read_jsonl
from .statistics import mcnemar_exact, wilson_interval


def pct(x: float | None) -> float | None:
    return None if x is None else 100.0 * x


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_all(discovered: DiscoveredArtifacts) -> dict[str, Any]:
    p = discovered.paths
    return {
        "model_compare": read_json(p["model_a_vs_a2_vs_a3"]),
        "a3_final": read_json(p["a3_final"]),
        "b_final": read_json(p["b_final"]),
        "a4": read_json(p["a4_terminal"]),
        "timing": read_json(p["timing_ablation"]),
        "timing_rows": read_json(p["timing_rows"]),
        "llm4": read_json(p["llm_qwen4_summary"]),
        "llm9": read_json(p["llm_qwen9_summary"]),
        "classic_freeze": read_json(p["classic_freeze"]),
        "classic_verification": read_json(p["classic_verification"]),
        "classic": {
            "a3": _load_classic_run(p["classic_a3"]),
            "a4": _load_classic_run(p["classic_a4"]),
            "qwen4": _load_classic_run(p["classic_qwen4"]),
            "qwen9": _load_classic_run(p["classic_qwen9"]),
        },
    }


def _load_classic_run(run_dir: Path) -> dict[str, Any]:
    return {
        "config": read_json(run_dir / "config.json"),
        "summary": read_json(run_dir / "summary.json"),
        "by_mate_depth": read_json(run_dir / "by_mate_depth.json"),
        "by_verification_status": read_json(run_dir / "by_verification_status.json"),
        "predictions": read_jsonl(run_dir / "predictions.jsonl"),
        "run_dir": str(run_dir),
    }


def validate(payload: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    a3 = payload["a3_final"]["test_metrics"]
    a4 = payload["a4"]["a4"]
    if a3["num_examples"] != 8610:
        raise ValueError(f"Lichess A3 N invariant failed: {a3['num_examples']}")
    if payload["a4"]["a3"]["n"] != 8610 or a4["n"] != 8610:
        raise ValueError("Lichess A4 shared N invariant failed")
    checks.append("Lichess A3/A4 N = 8610")

    freeze = payload["classic_freeze"]
    verification = payload["classic_verification"]
    if freeze["dataset_statistics"]["total"] != 200:
        raise ValueError("Classic total invariant failed")
    if set(freeze["dataset_statistics"]["mate_depth_distribution"].values()) != {20}:
        raise ValueError("Classic per-depth invariant failed")
    if freeze["freeze_fingerprint"] != "57ff2b7c725a4ef802817581fac9430f8c6416a6de54dafcb1deb7b471716b60":
        raise ValueError("Classic freeze fingerprint invariant failed")
    if (verification["verified_total"], verification["verified_unique_key"], verification["verified_multiple_keys_including_source"], verification["unresolved_timeout"]) != (193, 188, 5, 7):
        raise ValueError("Classic verification totals invariant failed")
    checks.append("Classic frozen dataset and verification totals validated")

    expected = {
        "a3": (22, 47, 69),
        "a4": (31, 59, 69),
    }
    for key, (top1, top3, top5) in expected.items():
        summary = payload["classic"][key]["summary"]["all_200"]
        actual = (summary["top1_correct"], round(summary["top3_accuracy"] * 200), round(summary["top5_accuracy"] * 200))
        if actual != (top1, top3, top5):
            raise ValueError(f"Classic {key} TopK invariant failed: {actual}")
    checks.append("Classic A3/A4 TopK counts validated")

    expected_qwen = {
        "qwen4": (0, 3, 101, 96, 0),
        "qwen9": (1, 6, 64, 129, 0),
    }
    for key, expected_counts in expected_qwen.items():
        counts = payload["classic"][key]["summary"]["diagnostics"]["qwen_outcome_counts"]
        actual = (counts["correct"], counts["wrong_legal"], counts["illegal"], counts["parse_failure"], counts["runtime_failure"])
        if actual != expected_counts:
            raise ValueError(f"Classic {key} Qwen invariant failed: {actual}")
        if sum(actual) != counts["total"]:
            raise ValueError(f"Classic {key} Qwen category sum failed")
    checks.append("Classic Qwen valid-attempt category sums validated")

    if payload["classic"]["a4"]["summary"]["all_200"]["top1_correct"] > round(payload["classic"]["a3"]["summary"]["all_200"]["top5_accuracy"] * 200):
        raise ValueError("A4 classic Top1 exceeded A3 Top5 retrieval ceiling")
    checks.append("A4 classic retrieval ceiling validated")
    return checks


def build_tables(discovered: DiscoveredArtifacts) -> dict[str, Any]:
    data = load_all(discovered)
    validation = validate(data)
    mc = data["model_compare"]["global"]
    a3 = data["a3_final"]["test_metrics"]
    b = data["b_final"]["test_metrics"]
    a4 = data["a4"]["a4"]
    llm4 = data["llm4"]
    llm9 = data["llm9"]

    lichess_primary = [
        _primary_row("A", "Global vocabulary GAT", "no", "no", "no", mc["n"], mc["raw_top1"], mc["raw_top3"], mc["raw_top5"], mc["model_a_mean_legal_rank"], mc["model_a_median_legal_rank"], mc["raw_illegal_top1_rate"], "Frozen no-timing baseline"),
        _primary_row("A1", "A + best-legal filter", "no", "filter only", "no", mc["n"], mc["best_legal_top1"], mc["best_legal_top3"], mc["best_legal_top5"], None, None, None, "Inference-only; no separate checkpoint"),
        _primary_row("A2", "Global vocabulary GAT with legal mask", "no", "yes", "no", mc["n"], mc["a2_masked_top1"], mc["a2_masked_top3"], mc["a2_masked_top5"], mc["model_a2_mean_legal_rank"], mc["model_a2_median_legal_rank"], mc["a2_masked_illegal_top1_rate"], "Legal class masking"),
        _primary_row("A3", "Legal-move candidate scorer", "no", "yes", "no", a3["num_examples"], a3["top1"], a3["top3"], a3["top5"], a3["mean_legal_target_rank"], a3["median_legal_target_rank"], a3["illegal_top1_rate"], "Matched no-timing baseline for RQ2"),
        _primary_row("B", "Timing-aware legal-move candidate scorer", "synthetic", "yes", "no", b["num_examples"], b["top1"], b["top3"], b["top5"], b["mean_legal_target_rank"], b["median_legal_target_rank"], b["illegal_top1_rate"], "Synthetic timing; not applicable to classic benchmark"),
        _primary_row("A4", "A3 Top-5 retrieval + post-move GNN reranker", "no", "yes", "yes", a4["n"], a4["end_to_end_top1"], None, None, None, None, 0.0, "Top1 only in terminal summary"),
        _primary_row("Qwen 3.5 4B strict", "LLM", "N/A", "N/A", "N/A", llm4["N"], llm4["top1_exact_canonical_accuracy"], None, None, None, None, None, "Strict UCI, one generated move"),
        _primary_row("Qwen 3.5 9B strict", "LLM", "N/A", "N/A", "N/A", llm9["N"], llm9["top1_exact_canonical_accuracy"], None, None, None, None, None, "Strict UCI, one generated move"),
    ]

    lichess_depth = _lichess_depth_rows(data)
    lichess_rating = _lichess_rating_rows(data)
    timing_ablation = _timing_ablation_rows(data)
    classic_primary = _classic_primary_rows(data)
    classic_depth = _classic_depth_rows(data)
    classic_verification = _classic_verification_rows(data)
    llm_failure = _llm_failure_rows(data)
    generalization = _generalization_rows(lichess_primary, classic_primary)
    stats = _statistical_tests(data, lichess_primary, classic_primary, classic_depth)
    rq1 = _rq1(data, lichess_depth)
    rq2 = _rq2(timing_ablation, stats)

    return {
        "validation": validation,
        "experiment_inventory": {"included": [r.to_json() for r in discovered.included], "excluded": [r.to_json() for r in discovered.excluded]},
        "source_manifest": {"included": [r.to_json() for r in discovered.included], "excluded": [r.to_json() for r in discovered.excluded]},
        "lichess_primary": lichess_primary,
        "lichess_by_mate_depth": lichess_depth,
        "lichess_by_rating": lichess_rating,
        "timing_ablation": timing_ablation,
        "classic_primary": classic_primary,
        "classic_by_mate_depth": classic_depth,
        "classic_by_verification": classic_verification,
        "llm_failure_analysis": llm_failure,
        "generalization": generalization,
        "statistical_tests": stats,
        "rq1_results": rq1,
        "rq2_results": rq2,
    }


def _primary_row(model, family, timing, legal, rerank, n, top1, top3, top5, mean_rank, median_rank, illegal, notes):
    correct = None if top1 is None or n is None else round(top1 * n)
    ci = None if correct is None else wilson_interval(correct, n)
    return {
        "model": model,
        "architecture_family": family,
        "timing": timing,
        "legal_candidate_restriction": legal,
        "post_move_reranking": rerank,
        "N": n,
        "Top1_percent": pct(top1),
        "Top3_percent": pct(top3),
        "Top5_percent": pct(top5),
        "mean_rank": mean_rank,
        "median_rank": median_rank,
        "illegal_Top1_rate_percent": pct(illegal),
        "Top1_Wilson95_low_percent": None if ci is None else pct(ci["lower"]),
        "Top1_Wilson95_high_percent": None if ci is None else pct(ci["upper"]),
        "protocol_notes": notes,
    }


def _lichess_depth_rows(data):
    rows = []
    a4_by = {r["bucket"].replace("MateIn", ""): r for r in data["a4"]["mate_depth_breakdown"]}
    timing_rows = data["timing_rows"]
    b_by = {}
    for depth in range(1, 6):
        subset = [r for r in timing_rows if int(r["MateDepth"]) == depth]
        if subset:
            b_by[str(depth)] = {
                "N": len(subset),
                "B": sum(r["B_SYNTHETIC_TIMING_top1_correct"] for r in subset) / len(subset),
            }
    for depth in range(1, 6):
        key = str(depth)
        a4 = a4_by.get(key)
        for model, acc, n in (
            ("A3", None if not a4 else a4["a3_top1"], None if not a4 else a4["n"]),
            ("B", b_by.get(key, {}).get("B"), b_by.get(key, {}).get("N")),
            ("A4", None if not a4 else a4["a4_end_to_end_top1"], None if not a4 else a4["n"]),
            ("Qwen 3.5 4B strict", data["llm4"]["accuracy_by_mate_depth"][key]["accuracy"], data["llm4"]["accuracy_by_mate_depth"][key]["N"]),
            ("Qwen 3.5 9B strict", data["llm9"]["accuracy_by_mate_depth"][key]["accuracy"], data["llm9"]["accuracy_by_mate_depth"][key]["N"]),
        ):
            rows.append({"model": model, "mate_depth": f"MateIn{depth}", "N": n, "Top1_percent": pct(acc)})
    return rows


def _lichess_rating_rows(data):
    rows = []
    for item in data["a4"]["rating_breakdown"]:
        rows.append({"model": "A3", "rating_bucket": item["bucket"], "N": item["n"], "Top1_percent": pct(item["a3_top1"])})
        rows.append({"model": "A4", "rating_bucket": item["bucket"], "N": item["n"], "Top1_percent": pct(item["a4_end_to_end_top1"])})
    return rows


def _timing_ablation_rows(data):
    perf = data["timing"]["official_model_performance"]
    rows = []
    for label, source, mode in (("A3", perf["A3"], "no timing"), ("B_SYNTHETIC_TIMING", perf["B_SYNTHETIC_TIMING"], "synthetic timing")):
        rows.append({"model": label, "timing_mode": mode, "N": source["n"], "Top1_percent": pct(source["top1"]), "Top3_percent": pct(source["top3"]), "Top5_percent": pct(source["top5"]), "mean_rank": source["mean_legal_target_rank"], "median_rank": source["median_legal_target_rank"]})
    for label, source in data["timing"]["post_hoc_diagnostic_ablations"].items():
        rows.append({"model": label, "timing_mode": "neutralized timing" if "NEUTRAL" in label else "zero timing", "N": source["n"], "Top1_percent": pct(source["top1"]), "Top3_percent": pct(source["top3"]), "Top5_percent": pct(source["top5"]), "mean_rank": source["mean_legal_target_rank"], "median_rank": source["median_legal_target_rank"]})
    a3_top1, b_top1 = rows[0]["Top1_percent"], rows[1]["Top1_percent"]
    for row in rows:
        row["delta_Top1_vs_A3_pp"] = None if row["Top1_percent"] is None else row["Top1_percent"] - a3_top1
    rows[1]["relative_Top1_change_vs_A3_percent"] = (b_top1 - a3_top1) / a3_top1 * 100
    return rows


def _classic_primary_rows(data):
    rows = []
    mapping = [("A3", "a3"), ("A4", "a4"), ("Qwen 3.5 4B strict", "qwen4"), ("Qwen 3.5 9B strict", "qwen9")]
    for label, key in mapping:
        run = data["classic"][key]["summary"]
        all200 = run["all_200"]
        qwen = run["diagnostics"].get("qwen_outcome_counts", {})
        ci = wilson_interval(all200["top1_correct"], all200["total"])
        rows.append({
            "model": label,
            "ALL200_Top1_percent": pct(all200["top1_accuracy"]),
            "Top3_percent": pct(all200.get("top3_accuracy")),
            "Top5_percent": pct(all200.get("top5_accuracy")),
            "correct_count": all200["top1_correct"],
            "Popeye193_Top1_percent": pct(run["popeye_verified_193"]["top1_accuracy"]),
            "mean_rank": all200.get("mean_accepted_key_rank"),
            "median_rank": all200.get("median_accepted_key_rank"),
            "parse_failure_count": qwen.get("parse_failure"),
            "illegal_count": qwen.get("illegal"),
            "wrong_legal_count": qwen.get("wrong_legal"),
            "runtime_failure_count": qwen.get("runtime_failure"),
            "Wilson95_low_percent": pct(ci["lower"]),
            "Wilson95_high_percent": pct(ci["upper"]),
        })
    rows.append({"model": "B", "ALL200_Top1_percent": None, "Top3_percent": None, "Top5_percent": None, "correct_count": None, "Popeye193_Top1_percent": None, "mean_rank": None, "median_rank": None, "parse_failure_count": None, "illegal_count": None, "wrong_legal_count": None, "runtime_failure_count": None, "notes": "N/A - TIMING_UNAVAILABLE"})
    return rows


def _classic_depth_rows(data):
    rows = []
    for label, key in [("A3", "a3"), ("A4", "a4"), ("Qwen 3.5 4B strict", "qwen4"), ("Qwen 3.5 9B strict", "qwen9")]:
        for depth in range(1, 11):
            item = data["classic"][key]["by_mate_depth"][str(depth)]
            ci = wilson_interval(item["top1_correct"], item["total"])
            rows.append({"model": label, "mate_depth": f"#{depth}", "N": item["total"], "Top1_percent": pct(item["top1_accuracy"]), "correct_count": item["top1_correct"], "Wilson95_low_percent": pct(ci["lower"]), "Wilson95_high_percent": pct(ci["upper"])})
    return rows


def _classic_verification_rows(data):
    rows = []
    for label, key in [("A3", "a3"), ("A4", "a4"), ("Qwen 3.5 4B strict", "qwen4"), ("Qwen 3.5 9B strict", "qwen9")]:
        for category, item in data["classic"][key]["by_verification_status"].items():
            rows.append({"model": label, "verification_category": category, "N": item["total"], "Top1_percent": pct(item["top1_accuracy"]), "correct_count": item["top1_correct"]})
    return rows


def _llm_failure_rows(data):
    rows = []
    for benchmark, label, summary in [("Lichess", "Qwen 3.5 4B strict", data["llm4"]), ("Lichess", "Qwen 3.5 9B strict", data["llm9"])]:
        counts = {"correct": summary["correct"], "wrong_legal": summary["wrong_legal"], "illegal": summary["illegal"], "parse_failure": summary["parse_errors"], "runtime_failure": summary["runtime_errors"]}
        for outcome, count in counts.items():
            rows.append({"benchmark": benchmark, "model": label, "outcome": outcome, "count": count, "percent": pct(count / summary["N"])})
    for label, key in [("Qwen 3.5 4B strict", "qwen4"), ("Qwen 3.5 9B strict", "qwen9")]:
        counts = data["classic"][key]["summary"]["diagnostics"]["qwen_outcome_counts"]
        for outcome in ("correct", "wrong_legal", "illegal", "parse_failure", "runtime_failure"):
            rows.append({"benchmark": "YACPDB classic", "model": label, "outcome": outcome, "count": counts[outcome], "percent": pct(counts[outcome] / counts["total"])})
    return rows


def _generalization_rows(lichess_primary, classic_primary):
    lookup_l = {r["model"]: r for r in lichess_primary}
    lookup_c = {r["model"]: r for r in classic_primary}
    pairs = [("A3", "A3"), ("A4", "A4"), ("Qwen 3.5 4B strict", "Qwen 3.5 4B strict"), ("Qwen 3.5 9B strict", "Qwen 3.5 9B strict")]
    rows = []
    for lkey, ckey in pairs:
        lval = lookup_l[lkey]["Top1_percent"]
        cval = lookup_c[ckey]["ALL200_Top1_percent"]
        rows.append({"model": lkey, "lichess_Top1_percent": lval, "classic_Top1_percent": cval, "classic_minus_lichess_pp": cval - lval})
    return rows


def _statistical_tests(data, lichess_primary, classic_primary, classic_depth):
    tests: dict[str, Any] = {"wilson_intervals": {"lichess_top1": {}, "classic_top1": {}, "classic_depth_top1": {}}, "paired_tests": {}}
    for row in lichess_primary:
        if row["Top1_percent"] is not None:
            tests["wilson_intervals"]["lichess_top1"][row["model"]] = {"low_percent": row["Top1_Wilson95_low_percent"], "high_percent": row["Top1_Wilson95_high_percent"], "N": row["N"]}
    for row in classic_primary:
        if row.get("ALL200_Top1_percent") is not None:
            tests["wilson_intervals"]["classic_top1"][row["model"]] = {"low_percent": row["Wilson95_low_percent"], "high_percent": row["Wilson95_high_percent"]}
    for row in classic_depth:
        tests["wilson_intervals"]["classic_depth_top1"][f"{row['model']} {row['mate_depth']}"] = {"low_percent": row["Wilson95_low_percent"], "high_percent": row["Wilson95_high_percent"], "N": row["N"]}
    tr = data["timing_rows"]
    tests["paired_tests"]["Lichess_A3_vs_B"] = mcnemar_exact([r["A3_top1_correct"] for r in tr], [r["B_SYNTHETIC_TIMING_top1_correct"] for r in tr])
    a3p = {r["heldout_id"]: r["is_correct"] for r in data["classic"]["a3"]["predictions"]}
    a4p = {r["heldout_id"]: r["is_correct"] for r in data["classic"]["a4"]["predictions"]}
    ids = sorted(set(a3p) & set(a4p))
    tests["paired_tests"]["Classic_A3_vs_A4"] = mcnemar_exact([a3p[i] for i in ids], [a4p[i] for i in ids])
    tests["effect_sizes"] = {
        "A3_vs_B_Top1_delta_pp": next(r for r in data["timing"]["official_model_performance"]["shared_comparison_frame"] if r["metric"] == "Top1")["delta_b_minus_a3_pp"],
        "Classic_A4_conditional_reranking_success_percent": pct(31 / 69),
    }
    tests["warnings"] = ["Classic mate-depth buckets have N=20; confidence intervals are wide.", "No large family of post-hoc significance tests is generated."]
    return tests


def _rq1(data, depth_rows):
    rows = []
    by = {(r["model"], r["mate_depth"]): r for r in depth_rows}
    for depth in range(1, 6):
        key = f"MateIn{depth}"
        b = by[("B", key)]
        q4 = by[("Qwen 3.5 4B strict", key)]
        q9 = by[("Qwen 3.5 9B strict", key)]
        rows.append({"mate_depth": key, "N": b["N"], "B_Top1_percent": b["Top1_percent"], "Qwen4_Top1_percent": q4["Top1_percent"], "Qwen9_Top1_percent": q9["Top1_percent"], "B_minus_Qwen4_pp": b["Top1_percent"] - q4["Top1_percent"], "B_minus_Qwen9_pp": b["Top1_percent"] - q9["Top1_percent"]})
    return {"status": "COMPUTED_LICHESS_ONLY", "note": "YACPDB is excluded from timed-GNN-vs-LLM comparison because timing is unavailable.", "rows": rows}


def _rq2(timing_ablation, stats):
    a3 = next(r for r in timing_ablation if r["model"] == "A3")
    b = next(r for r in timing_ablation if r["model"] == "B_SYNTHETIC_TIMING")
    return {
        "status": "COMPUTED",
        "controlled_comparison": "A3 no timing vs Model B synthetic timing",
        "top1_delta_b_minus_a3_pp": b["delta_Top1_vs_A3_pp"],
        "top3_delta_b_minus_a3_pp": b["Top3_percent"] - a3["Top3_percent"],
        "top5_delta_b_minus_a3_pp": b["Top5_percent"] - a3["Top5_percent"],
        "paired_test": stats["paired_tests"]["Lichess_A3_vs_B"],
        "bounded_conclusion": "The implemented synthetic timing signal did not improve predictive performance relative to the matched no-timing A3 baseline.",
    }


def write_outputs(out_dir: Path, tables: dict[str, Any]) -> None:
    data_dir = out_dir / "data"
    write_json(out_dir / "source_manifest.json", tables["source_manifest"])
    for name in ("experiment_inventory", "statistical_tests", "rq1_results", "rq2_results"):
        write_json(data_dir / f"{name}.json", tables[name])
    for name in ("lichess_primary", "lichess_by_mate_depth", "lichess_by_rating", "timing_ablation", "classic_primary", "classic_by_mate_depth", "classic_by_verification", "llm_failure_analysis", "generalization"):
        write_csv(data_dir / f"{name}.csv", tables[name])
        write_json(data_dir / f"{name}.json", tables[name])

