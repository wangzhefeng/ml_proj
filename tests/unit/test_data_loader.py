from mlproj.data.loader import DatasetLoader


def test_loader_sklearn_classification():
    loader = DatasetLoader(random_state=7)
    cfg = {
        "source": {"type": "sklearn", "name": "wine"},
        "split": {"strategy": "random", "valid_size": 0.2, "test_size": 0.2},
    }
    ds = loader.load(cfg)
    assert len(ds.X_train) > 0
    assert ds.y_train is not None
    assert len(ds.X_train.columns) > 0
