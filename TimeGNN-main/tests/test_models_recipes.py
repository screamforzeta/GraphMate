import pandas as pd
import pytest


@pytest.fixture()
def sample_event_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [1, 2, 3, 1, 2, 3],
            "sequence": ["a", "a", "a", "b", "b", "b"],
            "event": ["x", "y", "x", "x", "y", "x"],
            "status": ["s", "s", "s", "s", "s", "s"],
            "sn1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


def _skip_if_no_pyg() -> None:
    pytest.importorskip("torch_geometric")


def _supports_edge_attr() -> bool:
    try:
        from torch_geometric.nn import GATConv
    except Exception:
        return False

    import inspect

    sig = inspect.signature(GATConv.propagate)
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD or param.name == "edge_attr"
        for param in sig.parameters.values()
    )


def test_gat_basic_recipe_runs(sample_event_df: pd.DataFrame):
    _skip_if_no_pyg()
    from timegnn.recipes import train_gat_basic

    result = train_gat_basic(
        sample_event_df,
        case_index="sequence",
        core_event="event",
        start_time_col="time",
        cat_col_event=["event"],
        num_col_event=[],
        seq_cols=["sn1"],
        cat_col_seq=[],
        num_col_seq=["sn1"],
        num_epochs=1,
        batch_size=2,
    )
    assert "model" in result
    assert "history" in result


def test_gat_status_emb_recipe_runs(sample_event_df: pd.DataFrame):
    _skip_if_no_pyg()
    from timegnn.recipes import train_gat_status_emb

    result = train_gat_status_emb(
        sample_event_df,
        case_index="sequence",
        core_event="event",
        start_time_col="time",
        status_col="status",
        cat_col_event=["event"],
        num_col_event=[],
        seq_cols=["sn1"],
        cat_col_seq=[],
        num_col_seq=["sn1"],
        num_epochs=1,
        batch_size=2,
    )
    assert "model" in result
    assert "history" in result


def test_gat_time_decay_recipe_runs(sample_event_df: pd.DataFrame):
    _skip_if_no_pyg()
    if not _supports_edge_attr():
        pytest.skip("PyG propagate does not accept edge_attr in this environment")
    from timegnn.recipes import train_gat_time_decay

    result = train_gat_time_decay(
        sample_event_df,
        case_index="sequence",
        core_event="event",
        start_time_col="time",
        cat_col_event=["event"],
        num_col_event=[],
        seq_cols=["sn1"],
        cat_col_seq=[],
        num_col_seq=["sn1"],
        num_epochs=1,
        batch_size=2,
    )
    assert "model" in result
    assert "history" in result


def test_gat_time_decay_status_recipe_runs(sample_event_df: pd.DataFrame):
    _skip_if_no_pyg()
    if not _supports_edge_attr():
        pytest.skip("PyG propagate does not accept edge_attr in this environment")
    from timegnn.recipes import train_gat_time_decay_status_emb

    result = train_gat_time_decay_status_emb(
        sample_event_df,
        case_index="sequence",
        core_event="event",
        start_time_col="time",
        status_col="status",
        cat_col_event=["event"],
        num_col_event=[],
        seq_cols=["sn1"],
        cat_col_seq=[],
        num_col_seq=["sn1"],
        num_epochs=1,
        batch_size=2,
    )
    assert "model" in result
    assert "history" in result


def test_prefix_gcn_recipe_runs(sample_event_df: pd.DataFrame):
    _skip_if_no_pyg()
    from timegnn.recipes import train_prefix_gcn

    result = train_prefix_gcn(
        sample_event_df,
        case_index="sequence",
        core_event="event",
        start_time_col="time",
        cat_col_event=["event"],
        num_col_event=[],
        cat_col_seq=[],
        num_col_seq=["sn1"],
        num_epochs=1,
        batch_size=2,
        prefix_size=2,
        stratify=False,
    )
    assert "model" in result
    assert "history" in result


def test_gat_outcome_recipe_runs(sample_event_df: pd.DataFrame):
    _skip_if_no_pyg()
    from timegnn.recipes import train_gat_outcome

    df = sample_event_df.copy()
    df["outcome"] = ["ok", "ok", "ok", "fail", "fail", "fail"]

    result = train_gat_outcome(
        df,
        case_index="sequence",
        core_event="event",
        start_time_col="time",
        outcome_col="outcome",
        cat_col_event=["event"],
        num_col_event=[],
        seq_cols=["sn1"],
        mode="gat_basic",
        num_epochs=1,
        batch_size=2,
    )
    assert "model" in result
    assert "history" in result
    assert "label_encoder" in result
    assert "classes" in result


def test_early_stopping_triggers():
    from timegnn.train.early_stopping import EarlyStopping

    es = EarlyStopping(patience=2, delta=0.0)
    assert not es(1.0)
    assert not es(0.9)
    assert not es(1.0)  # counter = 1
    assert es(1.0)      # counter = 2 → stop
    assert es.early_stop


def test_early_stopping_best_loss_updated():
    from timegnn.train.early_stopping import EarlyStopping

    es = EarlyStopping(patience=3)
    es(1.0)
    assert es.best_loss_updated
    es(0.5)
    assert es.best_loss_updated
    es(0.6)
    assert not es.best_loss_updated
