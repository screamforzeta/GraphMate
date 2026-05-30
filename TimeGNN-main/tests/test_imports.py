def test_imports() -> None:
    from timegnn import Config, Pipeline

    assert Config is not None
    assert Pipeline is not None
