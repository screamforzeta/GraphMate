import numpy as np
import pandas as pd

from timegnn.data import encode_event_prefix_label, encode_event_prefix


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence": ["a", "a", "a", "b", "b", "b"],
            "event": ["x", "y", "x", "x", "y", "x"],
            "ec1": ["c", "c", "c", "d", "d", "d"],
        }
    )


def test_encode_event_prefix_label_shapes():
    df = _sample_df()
    text_encode, event_encode, y_encode, text_size, output_size = encode_event_prefix_label(
        df,
        core_event="event",
        cat_col_event=["ec1"],
        num_col_event=[],
        case_index="sequence",
        prefix_size=2,
    )

    assert text_encode.ndim == 3
    assert event_encode.ndim == 3
    assert y_encode.ndim == 1
    assert text_size > 0
    assert output_size > 0


def test_encode_event_prefix_shapes():
    df = _sample_df()
    event_encode, y_encode, output_size = encode_event_prefix(
        df,
        core_event="event",
        cat_col_event=["ec1"],
        num_col_event=[],
        case_index="sequence",
        prefix_size=2,
    )

    assert event_encode.ndim == 3
    assert y_encode.ndim == 1
    assert output_size > 0
