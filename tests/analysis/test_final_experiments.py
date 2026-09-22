from pathlib import Path

from src.analysis.final_experiments.artifacts import discover
from src.analysis.final_experiments.statistics import mcnemar_exact, wilson_interval
from src.analysis.final_experiments.tables import build_tables


ROOT = Path(".")


def test_artifact_discovery_includes_official_and_excludes_invalid_and_smoke():
    discovered = discover(ROOT)
    included = {(r.model_id, r.benchmark, r.validity_state) for r in discovered.included}
    excluded_states = {r.validity_state for r in discovered.excluded}
    assert ("MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING", "YACPDB classic", "VALID") in included
    assert ("qwen3.5:4b", "YACPDB classic", "VALID") in included
    assert "INVALID_INFRASTRUCTURE_RUN" in excluded_states
    assert "SMOKE_TEST" in excluded_states


def test_attempt2_selection_and_scientific_run_id():
    discovered = discover(ROOT)
    qwen4 = [
        r for r in discovered.included
        if r.model_id == "qwen3.5:4b" and r.benchmark == "YACPDB classic"
    ][0]
    assert qwen4.execution_attempt == 2
    assert qwen4.scientific_run_id == "qwen3.5:4b_8bf8da8d39c7e331"
    assert qwen4.validity_state == "VALID"


def test_table_aggregation_and_invariants():
    tables = build_tables(discover(ROOT))
    assert "Classic Qwen valid-attempt category sums validated" in tables["validation"]
    classic = {r["model"]: r for r in tables["classic_primary"]}
    assert classic["A3"]["correct_count"] == 22
    assert classic["A4"]["correct_count"] == 31
    assert classic["Qwen 3.5 4B strict"]["parse_failure_count"] == 96
    assert classic["B"]["notes"] == "N/A - TIMING_UNAVAILABLE"


def test_mate_depth_and_popeye_aggregation():
    tables = build_tables(discover(ROOT))
    a3_depth = [r for r in tables["classic_by_mate_depth"] if r["model"] == "A3"]
    assert len(a3_depth) == 10
    assert all(r["N"] == 20 for r in a3_depth)
    classic = {r["model"]: r for r in tables["classic_primary"]}
    assert round(classic["A3"]["Popeye193_Top1_percent"], 3) == 11.399


def test_qwen_category_sums_and_llm_failures():
    tables = build_tables(discover(ROOT))
    for model in ("Qwen 3.5 4B strict", "Qwen 3.5 9B strict"):
        rows = [r for r in tables["llm_failure_analysis"] if r["benchmark"] == "YACPDB classic" and r["model"] == model]
        assert round(sum(r["percent"] for r in rows), 10) == 100.0


def test_a3_a4_retrieval_accounting_and_paired_stats():
    tables = build_tables(discover(ROOT))
    tests = tables["statistical_tests"]
    assert tests["effect_sizes"]["Classic_A4_conditional_reranking_success_percent"] == 31 / 69 * 100
    assert tests["paired_tests"]["Classic_A3_vs_A4"]["status"] == "COMPUTED"
    assert tests["paired_tests"]["Lichess_A3_vs_B"]["status"] == "COMPUTED"


def test_wilson_interval_and_mcnemar_helpers():
    ci = wilson_interval(22, 200)
    assert ci["lower"] < 0.11 < ci["upper"]
    result = mcnemar_exact([True, False, False], [False, True, False])
    assert result["discordant"] == 2
    assert result["status"] == "COMPUTED"

