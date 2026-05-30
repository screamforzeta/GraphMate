import pandas as pd
import pytest


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [1, 2, 3, 1, 2, 3],
            "sequence": ["a", "a", "a", "b", "b", "b"],
            "event": ["x", "y", "x", "x", "y", "x"],
            "status": ["s", "s", "s", "s", "s", "s"],
            "sn1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


def test_api_functions_exist():
    from timegnn import api

    assert hasattr(api, "gat_basic")
    assert hasattr(api, "gat_status")
    assert hasattr(api, "gat_time_decay")
    assert hasattr(api, "gat_time_decay_status")
    assert hasattr(api, "prefix_gcn")


def test_api_accepts_overrides():
    from timegnn.api import gat_basic

    df = _sample_df()
    pytest.importorskip("torch_geometric")

    result = gat_basic(
        df,
        case_col="sequence",
        event_col="event",
        time_col="time",
        cat_event=["event"],
        num_event=[],
        seq_cols=["sn1"],
        cat_seq=[],
        num_seq=["sn1"],
        num_layers=2,
        dropout=0.1,
        use_batch_norm=True,
        activation="gelu",
        num_epochs=1,
        patience=1,
        delta=0.0,
        batch_size=2,
    )
    assert "model" in result
    assert "history" in result
