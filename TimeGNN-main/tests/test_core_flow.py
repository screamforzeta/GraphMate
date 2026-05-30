import pandas as pd

from timegnn import Config, Pipeline
from timegnn.data import EventSchema
from timegnn.encoders import BasicLabelEncoder
from timegnn.models.baseline import BaselineMostFrequentModel
from timegnn.train import BasicTrainer


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [1, 2, 3, 1, 2],
            "sequence": ["a", "a", "a", "b", "b"],
            "event": ["x", "y", "x", "x", "x"],
        }
    )


def test_event_schema_validation_passes() -> None:
    df = _sample_df()
    schema = EventSchema(time_col="time", case_col="sequence", event_col="event")
    schema.validate(df)


def test_event_schema_validation_fails() -> None:
    df = _sample_df().drop(columns=["event"])
    schema = EventSchema(time_col="time", case_col="sequence", event_col="event")
    try:
        schema.validate(df)
    except ValueError as exc:
        assert "Missing required columns" in str(exc)
    else:
        raise AssertionError("Expected schema validation to fail.")


def test_basic_label_encoder_roundtrip() -> None:
    df = _sample_df()
    encoder = BasicLabelEncoder()
    encoder.fit(df["event"])
    encoded = encoder.transform(df["event"])
    decoded = encoder.inverse_transform(encoded)
    assert list(decoded) == list(df["event"].astype(str))


def test_baseline_model_predicts_most_frequent() -> None:
    df = _sample_df()
    schema = EventSchema(time_col="time", case_col="sequence", event_col="event")
    encoder = BasicLabelEncoder()
    encoder.fit(df[schema.event_col])
    model = BaselineMostFrequentModel()
    model.fit(df, schema, encoder)
    preds = model.predict(df, schema, encoder)
    assert preds.nunique() == 1


def test_trainer_evaluate_returns_accuracy() -> None:
    df = _sample_df()
    schema = EventSchema(time_col="time", case_col="sequence", event_col="event")
    encoder = BasicLabelEncoder()
    encoder.fit(df[schema.event_col])
    model = BaselineMostFrequentModel()
    trainer = BasicTrainer()
    trainer.train(model, df, schema, encoder)
    metrics = trainer.evaluate(model, df, schema, encoder)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_pipeline_end_to_end(tmp_path) -> None:
    df = _sample_df()
    csv_path = tmp_path / "events.csv"
    df.to_csv(csv_path, index=False)

    cfg = Config(
        model="baseline_most_frequent",
        task="next_event",
        data_source=str(csv_path),
        time_col="time",
        case_col="sequence",
        event_col="event",
    )

    pipe = Pipeline(cfg)
    pipe.fit()
    metrics = pipe.evaluate()
    assert "accuracy" in metrics
