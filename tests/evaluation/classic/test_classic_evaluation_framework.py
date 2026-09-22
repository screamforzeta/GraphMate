import json
from pathlib import Path

import pytest

from src.evaluation.classic import (
    A3ClassicAdapter,
    A4ClassicAdapter,
    EXPECTED_FREEZE_FINGERPRINT,
    ModelBClassicAdapter,
    MockClassicAdapter,
    QwenClassicAdapter,
    aggregate_predictions,
    build_prediction_record,
    build_run_identity,
    load_classic_samples,
    prepare_official_run_directory,
    score_classic_prediction,
    verify_frozen_benchmark,
    write_prediction_records,
)
from src.evaluation.classic import core as classic_core
from src.cli.evaluation import evaluate_classic_heldout as classic_cli
from src.evaluation.classic.runner import mark_completed
from src.verification.popeye_consolidate import ACCEPTED_KEY_POLICY_VERSION, EXPECTED_DATASET_FINGERPRINT


FEN = "k7/8/8/8/8/8/4K3/8 w - - 0 1"


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def make_fixture(root: Path):
    dataset = root / "dataset"
    consolidated = root / "consolidated"
    canonical = [
        {
            "heldout_id": "fixture_001",
            "source_problem_id": "p1",
            "mate_depth": 1,
            "fen": FEN,
            "key_move_uci": "e2e3",
        },
        {
            "heldout_id": "fixture_002",
            "source_problem_id": "p2",
            "mate_depth": 2,
            "fen": FEN,
            "key_move_uci": "e2d2",
        },
        {
            "heldout_id": "fixture_003",
            "source_problem_id": "p3",
            "mate_depth": 2,
            "fen": FEN,
            "key_move_uci": "e2f2",
        },
    ]
    results = [
        {
            "heldout_id": "fixture_001",
            "source_problem_id": "p1",
            "mate_depth": 1,
            "canonical_fen": FEN,
            "source_key_move_uci": "e2e3",
            "accepted_key_moves_uci": ["e2e3"],
            "accepted_key_basis": "POPEYE_VERIFIED_UNIQUE",
            "final_verification_status": "VERIFIED",
            "final_verification_reason": "VERIFIED_UNIQUE_KEY_MATCH",
            "forced_mate_verified": True,
            "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        },
        {
            "heldout_id": "fixture_002",
            "source_problem_id": "p2",
            "mate_depth": 2,
            "canonical_fen": FEN,
            "source_key_move_uci": "e2d2",
            "accepted_key_moves_uci": ["e2d2", "e2e3"],
            "accepted_key_basis": "POPEYE_VERIFIED_MULTIPLE",
            "final_verification_status": "VERIFIED",
            "final_verification_reason": "VERIFIED_MULTIPLE_KEYS_INCLUDES_SOURCE_KEY",
            "forced_mate_verified": True,
            "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        },
        {
            "heldout_id": "fixture_003",
            "source_problem_id": "p3",
            "mate_depth": 2,
            "canonical_fen": FEN,
            "source_key_move_uci": "e2f2",
            "accepted_key_moves_uci": ["e2f2"],
            "accepted_key_basis": "YACPDB_SOURCE_UNVERIFIED_TIMEOUT",
            "final_verification_status": "VERIFICATION_INCONCLUSIVE_TIMEOUT",
            "final_verification_reason": "TIMEOUT",
            "forced_mate_verified": False,
            "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        },
    ]
    accepted = [
        {
            "heldout_id": row["heldout_id"],
            "source_problem_id": row["source_problem_id"],
            "mate_depth": row["mate_depth"],
            "source_key_move_uci": row["source_key_move_uci"],
            "accepted_key_moves_uci": row["accepted_key_moves_uci"],
            "accepted_key_basis": row["accepted_key_basis"],
        }
        for row in results
    ]
    write_json(dataset / "manifest.json", {"dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT})
    write_jsonl(dataset / "dataset.jsonl", canonical)
    write_jsonl(consolidated / "accepted_keys.jsonl", accepted)
    write_jsonl(consolidated / "results.jsonl", results)
    return dataset, consolidated


def make_200_fixture(root: Path):
    dataset = root / "dataset_200"
    consolidated = root / "consolidated_200"
    canonical = []
    results = []
    for index in range(200):
        depth = index % 10 + 1
        heldout_id = f"fixture_200_{index:04d}"
        forced = index < 193
        basis = (
            "POPEYE_VERIFIED_UNIQUE"
            if index < 188
            else "POPEYE_VERIFIED_MULTIPLE"
            if index < 193
            else "YACPDB_SOURCE_UNVERIFIED_TIMEOUT"
        )
        canonical.append(
            {
                "heldout_id": heldout_id,
                "source_problem_id": f"p{index}",
                "mate_depth": depth,
                "fen": FEN,
                "key_move_uci": "e2e3",
            }
        )
        results.append(
            {
                "heldout_id": heldout_id,
                "source_problem_id": f"p{index}",
                "mate_depth": depth,
                "canonical_fen": FEN,
                "source_key_move_uci": "e2e3",
                "accepted_key_moves_uci": ["e2e3"],
                "accepted_key_basis": basis,
                "final_verification_status": "VERIFIED" if forced else "VERIFICATION_INCONCLUSIVE_TIMEOUT",
                "final_verification_reason": "VERIFIED_UNIQUE_KEY_MATCH" if forced else "TIMEOUT",
                "forced_mate_verified": forced,
                "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
            }
        )
    accepted = [
        {
            "heldout_id": row["heldout_id"],
            "source_problem_id": row["source_problem_id"],
            "mate_depth": row["mate_depth"],
            "source_key_move_uci": row["source_key_move_uci"],
            "accepted_key_moves_uci": row["accepted_key_moves_uci"],
            "accepted_key_basis": row["accepted_key_basis"],
        }
        for row in results
    ]
    write_json(dataset / "manifest.json", {"dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT})
    write_jsonl(dataset / "dataset.jsonl", canonical)
    write_jsonl(consolidated / "accepted_keys.jsonl", accepted)
    write_jsonl(consolidated / "results.jsonl", results)
    return dataset, consolidated


def test_sample_loader_joins_canonical_and_accepted_keys(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)

    samples = load_classic_samples(dataset, consolidated, require_frozen=False)

    assert [sample.heldout_id for sample in samples] == ["fixture_001", "fixture_002", "fixture_003"]
    assert samples[1].accepted_key_moves_uci == ("e2d2", "e2e3")
    assert samples[2].verification_category == "unresolved_timeout_source_only"


def test_schema_mapping_allows_different_field_names_with_same_values(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)

    samples = load_classic_samples(dataset, consolidated, require_frozen=False)

    assert samples[0].fen == FEN
    assert samples[0].source_key_move_uci == "e2e3"


def test_evaluator_rejects_non_frozen_benchmark(monkeypatch):
    def fake_verify(*args, **kwargs):
        return {"lifecycle": "VALIDATED_NOT_FROZEN", "freeze_fingerprint": EXPECTED_FREEZE_FINGERPRINT}

    monkeypatch.setattr(classic_core, "verify_freeze_manifest", fake_verify)

    with pytest.raises(RuntimeError, match="lifecycle"):
        verify_frozen_benchmark()


def test_evaluator_rejects_wrong_freeze_fingerprint(monkeypatch):
    def fake_verify(*args, **kwargs):
        return {
            "lifecycle": "FROZEN",
            "freeze_fingerprint": "wrong",
            "verification": {"accepted_key_policy_version": ACCEPTED_KEY_POLICY_VERSION},
        }

    monkeypatch.setattr(classic_core, "verify_freeze_manifest", fake_verify)

    with pytest.raises(RuntimeError, match="fingerprint"):
        verify_frozen_benchmark()


def test_sample_loader_fails_closed_on_disagreement(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    rows = [json.loads(line) for line in (consolidated / "results.jsonl").read_text().splitlines()]
    rows[0]["canonical_fen"] = "8/8/8/8/8/8/4K3/4k3 b - - 0 1"
    write_jsonl(consolidated / "results.jsonl", rows)

    with pytest.raises(RuntimeError, match="fen->canonical_fen"):
        load_classic_samples(dataset, consolidated, require_frozen=False)


def test_sample_loader_fails_closed_on_source_key_source_id_and_depth_disagreement(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    rows = [json.loads(line) for line in (consolidated / "results.jsonl").read_text().splitlines()]

    rows[0]["source_key_move_uci"] = "e2d3"
    write_jsonl(consolidated / "results.jsonl", rows)
    with pytest.raises(RuntimeError, match="key_move_uci->source_key_move_uci"):
        load_classic_samples(dataset, consolidated, require_frozen=False)

    dataset, consolidated = make_fixture(tmp_path / "id")
    rows = [json.loads(line) for line in (consolidated / "results.jsonl").read_text().splitlines()]
    rows[0]["source_problem_id"] = "different"
    write_jsonl(consolidated / "results.jsonl", rows)
    with pytest.raises(RuntimeError, match="source_problem_id->source_problem_id"):
        load_classic_samples(dataset, consolidated, require_frozen=False)

    dataset, consolidated = make_fixture(tmp_path / "depth")
    rows = [json.loads(line) for line in (consolidated / "results.jsonl").read_text().splitlines()]
    rows[0]["mate_depth"] = 9
    write_jsonl(consolidated / "results.jsonl", rows)
    with pytest.raises(RuntimeError, match="mate_depth->mate_depth"):
        load_classic_samples(dataset, consolidated, require_frozen=False)


def test_sample_loader_fails_on_missing_duplicate_and_result_inconsistency(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    rows = [json.loads(line) for line in (consolidated / "accepted_keys.jsonl").read_text().splitlines()]
    write_jsonl(consolidated / "accepted_keys.jsonl", rows[:-1])
    with pytest.raises(RuntimeError, match="IDs disagree"):
        load_classic_samples(dataset, consolidated, require_frozen=False)

    dataset, consolidated = make_fixture(tmp_path / "dup")
    rows = [json.loads(line) for line in (consolidated / "accepted_keys.jsonl").read_text().splitlines()]
    write_jsonl(consolidated / "accepted_keys.jsonl", rows + [rows[0]])
    with pytest.raises(RuntimeError, match="Duplicate heldout_id"):
        load_classic_samples(dataset, consolidated, require_frozen=False)

    dataset, consolidated = make_fixture(tmp_path / "inconsistent")
    accepted_rows = [json.loads(line) for line in (consolidated / "accepted_keys.jsonl").read_text().splitlines()]
    result_rows = [json.loads(line) for line in (consolidated / "results.jsonl").read_text().splitlines()]
    result_rows[0]["accepted_key_moves_uci"] = ["e2d2"]
    write_jsonl(consolidated / "accepted_keys.jsonl", accepted_rows)
    write_jsonl(consolidated / "results.jsonl", result_rows)
    with pytest.raises(RuntimeError, match="Accepted-key/result inconsistency"):
        load_classic_samples(dataset, consolidated, require_frozen=False)


def test_projection_schema_does_not_need_canonical_fen_and_checks_projection_fields(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    accepted_rows = [json.loads(line) for line in (consolidated / "accepted_keys.jsonl").read_text().splitlines()]
    assert "canonical_fen" not in accepted_rows[0]

    samples = load_classic_samples(dataset, consolidated, require_frozen=False)
    assert samples[0].fen == FEN

    accepted_rows[0]["source_key_move_uci"] = "e2d3"
    write_jsonl(consolidated / "accepted_keys.jsonl", accepted_rows)
    with pytest.raises(RuntimeError, match="source_key_move_uci"):
        load_classic_samples(dataset, consolidated, require_frozen=False)

    dataset, consolidated = make_fixture(tmp_path / "accepted_moves")
    accepted_rows = [json.loads(line) for line in (consolidated / "accepted_keys.jsonl").read_text().splitlines()]
    accepted_rows[0]["accepted_key_moves_uci"] = ["e2d2"]
    write_jsonl(consolidated / "accepted_keys.jsonl", accepted_rows)
    with pytest.raises(RuntimeError, match="accepted_key_moves_uci"):
        load_classic_samples(dataset, consolidated, require_frozen=False)

    dataset, consolidated = make_fixture(tmp_path / "basis")
    accepted_rows = [json.loads(line) for line in (consolidated / "accepted_keys.jsonl").read_text().splitlines()]
    accepted_rows[0]["accepted_key_basis"] = "YACPDB_SOURCE_UNVERIFIED_TIMEOUT"
    write_jsonl(consolidated / "accepted_keys.jsonl", accepted_rows)
    with pytest.raises(RuntimeError, match="accepted_key_basis"):
        load_classic_samples(dataset, consolidated, require_frozen=False)


def test_missing_and_duplicate_results_fail(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    result_rows = [json.loads(line) for line in (consolidated / "results.jsonl").read_text().splitlines()]
    write_jsonl(consolidated / "results.jsonl", result_rows[:-1])
    with pytest.raises(RuntimeError, match="result IDs disagree"):
        load_classic_samples(dataset, consolidated, require_frozen=False)

    dataset, consolidated = make_fixture(tmp_path / "dup_results")
    result_rows = [json.loads(line) for line in (consolidated / "results.jsonl").read_text().splitlines()]
    write_jsonl(consolidated / "results.jsonl", result_rows + [result_rows[0]])
    with pytest.raises(RuntimeError, match="Duplicate heldout_id"):
        load_classic_samples(dataset, consolidated, require_frozen=False)


def test_final_forced_mate_verified_comes_from_results_not_canonical(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    canonical_rows = [json.loads(line) for line in (dataset / "dataset.jsonl").read_text().splitlines()]
    canonical_rows[0]["forced_mate_verified"] = False
    canonical_rows[0]["forced_mate_verification_status"] = "NOT_VERIFIED_ENGINE_NOT_USED"
    write_jsonl(dataset / "dataset.jsonl", canonical_rows)

    samples = load_classic_samples(dataset, consolidated, require_frozen=False)

    assert samples[0].forced_mate_verified is True


def test_all_200_structural_loader_validation_no_model_inference(tmp_path, monkeypatch):
    dataset, consolidated = make_200_fixture(tmp_path)

    def fail_if_model_inference(*args, **kwargs):
        raise AssertionError("model inference must not run during metadata loading")

    monkeypatch.setattr(A3ClassicAdapter, "predict", fail_if_model_inference)
    samples = load_classic_samples(dataset, consolidated, require_frozen=False)

    assert len(samples) == 200
    assert all(sum(sample.mate_depth == depth for sample in samples) == 20 for depth in range(1, 11))
    assert sum(sample.forced_mate_verified for sample in samples) == 193
    assert sum(not sample.forced_mate_verified for sample in samples) == 7
    assert sum(sample.accepted_key_basis == "POPEYE_VERIFIED_UNIQUE" for sample in samples) == 188
    assert sum(sample.accepted_key_basis == "POPEYE_VERIFIED_MULTIPLE" for sample in samples) == 5
    assert sum(sample.accepted_key_basis == "YACPDB_SOURCE_UNVERIFIED_TIMEOUT" for sample in samples) == 7


def test_metadata_only_validation_does_not_create_predictions(monkeypatch, tmp_path, capsys):
    dataset, consolidated = make_200_fixture(tmp_path)
    manifest = dataset / "freeze_manifest.json"
    write_json(manifest, {"lifecycle": "FROZEN", "freeze_fingerprint": EXPECTED_FREEZE_FINGERPRINT, "verification": {"accepted_key_policy_version": ACCEPTED_KEY_POLICY_VERSION}})

    monkeypatch.setattr(classic_cli, "verify_frozen_benchmark", lambda *args, **kwargs: {"lifecycle": "FROZEN", "freeze_fingerprint": EXPECTED_FREEZE_FINGERPRINT})
    monkeypatch.setattr(classic_core, "verify_frozen_benchmark", lambda *args, **kwargs: {"lifecycle": "FROZEN", "freeze_fingerprint": EXPECTED_FREEZE_FINGERPRINT})
    monkeypatch.setattr(
        classic_cli,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "official": False,
                "smoke_test": False,
                "validate_benchmark_only": True,
                "dataset_dir": str(dataset),
                "consolidated_dir": str(consolidated),
                "freeze_manifest": str(manifest),
            },
        )(),
    )

    assert classic_cli.main() == 0
    output = json.loads(capsys.readouterr().out)

    assert output["total"] == 200
    assert output["forced_mate_verified_true"] == 193
    assert not list(tmp_path.glob("**/predictions.jsonl"))


def test_unique_multi_key_and_timeout_scoring(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    samples = load_classic_samples(dataset, consolidated, require_frozen=False)

    assert score_classic_prediction("e2e3", samples[0].accepted_key_moves_uci)
    assert score_classic_prediction("e2d2", samples[1].accepted_key_moves_uci)
    assert score_classic_prediction("e2e3", samples[1].accepted_key_moves_uci)
    assert not score_classic_prediction("e2f2", samples[1].accepted_key_moves_uci)
    assert score_classic_prediction("e2f2", samples[2].accepted_key_moves_uci)
    assert samples[2].forced_mate_verified is False


def test_aggregation_required_strata(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    samples = load_classic_samples(dataset, consolidated, require_frozen=False)
    metadata = {"model_id": "mock", "model_family": "fixture", "model_version": "v"}
    records = [
        build_prediction_record(samples[0], metadata, "e2e3", ranked_moves_uci=["e2e3"]),
        build_prediction_record(samples[1], metadata, "e2e3", ranked_moves_uci=["e2d1", "e2e3"]),
        build_prediction_record(samples[2], metadata, "e2d1", ranked_moves_uci=["e2d1", "e2d2"]),
    ]

    summary = aggregate_predictions(records)

    assert summary["all_200"]["total"] == 3
    assert summary["all_200"]["top1_correct"] == 2
    assert summary["popeye_verified_193"]["total"] == 2
    assert summary["by_mate_depth"]["1"]["top1_correct"] == 1
    assert summary["by_mate_depth"]["2"]["total"] == 2
    assert summary["by_verification_category"]["YACPDB_SOURCE_UNVERIFIED_TIMEOUT"]["total"] == 1


def test_a3_adapter_uses_canonical_fen_without_lichess_moves0(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    sample = load_classic_samples(dataset, consolidated, require_frozen=False)[0]
    adapter = A3ClassicAdapter()

    metadata = adapter.metadata()
    candidates = adapter.candidates_for_sample(sample)

    assert metadata["applies_lichess_moves0"] is False
    assert "e2e3" in candidates


def test_a4_retrieval_receives_no_target_or_accepted_keys(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    sample = load_classic_samples(dataset, consolidated, require_frozen=False)[1]
    adapter = A4ClassicAdapter()

    payload = adapter.retrieval_inputs(sample)

    assert payload["fen"] == sample.fen
    assert payload["accepted_key_moves_uci"] is None
    assert payload["source_key_move_uci"] is None
    assert adapter.metadata()["retrieval_uses_target_or_accepted_keys"] is False


def test_a4_retrieval_and_reranking_failure_semantics():
    accepted = {"a1a2", "b1b2"}

    top5 = ["b1b2", "c1c2"]
    ranked = ["c1c2", "b1b2"]
    retrieval_success = bool(accepted.intersection(top5))
    reranking_failure = bool(retrieval_success and ranked[0] not in accepted)

    assert retrieval_success is True
    assert reranking_failure is True

    top5 = ["c1c2", "d1d2"]
    ranked = ["c1c2", "d1d2"]
    retrieval_success = bool(accepted.intersection(top5))
    reranking_failure = bool(retrieval_success and ranked[0] not in accepted)

    assert retrieval_success is False
    assert reranking_failure is False


def test_model_b_timing_protocol_explicit_and_no_random_generation(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    sample = load_classic_samples(dataset, consolidated, require_frozen=False)[0]
    adapter = ModelBClassicAdapter()
    metadata = adapter.metadata()

    assert metadata["timing_data_available"] is False
    assert metadata["timing_protocol"] == "not_applicable_timing_unavailable"
    assert metadata["random_timing_generation"] is False
    assert adapter.timing_values_for_sample(sample) is None
    assert metadata["timing_required_fields"] == [
        "previous_move_time",
        "original_move_time",
        "time_is_synthetic",
    ]


def test_model_b_cli_request_rejected_before_predictions(tmp_path):
    adapter = classic_cli.build_adapter(
        type(
            "Args",
            (),
            {
                "model": "MODEL_B_TIMING_LEGAL_SCORER",
                "a3_checkpoint": "unused",
                "a4_checkpoint": "unused",
                "device": "cpu",
                "amp": False,
                "ollama_url": None,
                "timeout": 1,
            },
        )()
    )

    with pytest.raises(RuntimeError, match="not applicable"):
        adapter.prepare()

    assert not list(tmp_path.glob("**/predictions.jsonl"))


def test_qwen_prompt_has_fen_only_and_parser_is_strict(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    sample = load_classic_samples(dataset, consolidated, require_frozen=False)[0]
    adapter = QwenClassicAdapter("qwen3.5:4b")

    prompt = adapter.build_prompt_for_sample(sample)
    parsed = adapter.parse_response_for_sample(sample, "e2e3")
    verbose = adapter.parse_response_for_sample(sample, "The move is e2e3.")

    assert sample.fen in prompt["user"]
    assert "MateDepth" not in prompt["user"]
    assert sample.source_key_move_uci not in prompt["user"].replace(sample.fen, "")
    assert prompt["version"] == "llm_chess_uci_v1"
    assert parsed.parse_success is True
    assert verbose.parse_success is False


def test_official_run_requires_flag_and_completed_run_is_not_overwritten(tmp_path):
    metadata = MockClassicAdapter({}).metadata()

    with pytest.raises(RuntimeError, match="--official"):
        prepare_official_run_directory(metadata, official=False, output_root=tmp_path)

    run_dir = prepare_official_run_directory(metadata, official=True, output_root=tmp_path)
    mark_completed(run_dir)

    with pytest.raises(RuntimeError, match="already completed"):
        prepare_official_run_directory(metadata, official=True, output_root=tmp_path)


def test_cli_requires_exactly_one_mode(monkeypatch):
    monkeypatch.setattr(
        classic_cli,
        "parse_args",
        lambda: type(
            "Args",
            (),
                {
                    "official": False,
                    "smoke_test": False,
                    "validate_benchmark_only": False,
                    "model": "a3",
                },
        )(),
    )

    with pytest.raises(RuntimeError, match="exactly one"):
        classic_cli.main()


def test_official_validation_verifies_freeze_before_prepare(monkeypatch, tmp_path):
    calls = []

    def fake_verify(*args, **kwargs):
        calls.append("verify")
        raise RuntimeError("freeze failed")

    class Adapter:
        def prepare(self):
            calls.append("prepare")

    args = type(
        "Args",
        (),
        {
            "dataset_dir": str(tmp_path),
            "consolidated_dir": str(tmp_path),
            "freeze_manifest": str(tmp_path / "freeze_manifest.json"),
            "expected_a3_sha256": None,
            "expected_a4_sha256": None,
        },
    )()
    monkeypatch.setattr(classic_cli, "verify_frozen_benchmark", fake_verify)

    with pytest.raises(RuntimeError, match="freeze failed"):
        classic_cli.validate_official_preconditions(args, Adapter())

    assert calls == ["verify"]


def test_wrong_checkpoint_hash_fails_before_prediction(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    adapter = A3ClassicAdapter(checkpoint)
    args = type(
        "Args",
        (),
        {
            "a3_checkpoint": str(checkpoint),
            "a4_checkpoint": str(checkpoint),
            "expected_a3_sha256": "not-the-real-hash",
            "expected_a4_sha256": None,
        },
    )()

    with pytest.raises(RuntimeError, match="SHA256"):
        classic_cli.validate_checkpoint_hashes(args, adapter)


def test_resume_does_not_duplicate_predictions_and_summary_from_records(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    samples = load_classic_samples(dataset, consolidated, require_frozen=False)
    metadata = MockClassicAdapter({}).metadata()
    record = build_prediction_record(samples[0], metadata, "e2e3")
    path = tmp_path / "predictions.jsonl"

    write_prediction_records(path, [record])
    with pytest.raises(RuntimeError, match="Duplicate"):
        write_prediction_records(path, [record, record])


def test_qwen_set_valued_scoring_after_strict_parse(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    sample = load_classic_samples(dataset, consolidated, require_frozen=False)[1]
    adapter = QwenClassicAdapter("qwen3.5:4b")

    parsed = adapter.parse_response_for_sample(sample, "e2e3")

    assert parsed.parse_success is True
    assert score_classic_prediction(parsed.parsed_uci, sample.accepted_key_moves_uci)


def test_run_identity_has_no_performance_metrics():
    metadata = {
        "model_id": "MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING",
        "model_family": "gnn_a3",
        "model_version": "frozen",
        "checkpoint_sha256": "abc",
    }

    identity = build_run_identity(metadata)

    assert identity["freeze_fingerprint"] == EXPECTED_FREEZE_FINGERPRINT
    assert "accuracy" not in json.dumps(identity).lower()
    assert identity["accepted_key_policy_version"] == ACCEPTED_KEY_POLICY_VERSION


def test_fixture_mode_does_not_use_frozen_heldout_ids(tmp_path):
    dataset, consolidated = make_fixture(tmp_path)
    samples = load_classic_samples(dataset, consolidated, require_frozen=False)

    assert all(not sample.heldout_id.startswith("yacpdb_classic_v1_") for sample in samples)
