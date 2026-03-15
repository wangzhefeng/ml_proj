import pandas as pd

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


def test_loader_csv_explicit_splits(tmp_path):
    train = pd.DataFrame({"f1": [1, 2, 3], "f2": [3, 4, 5], "target": [0, 1, 0]})
    valid = pd.DataFrame({"f1": [6, 7], "f2": [8, 9], "target": [1, 0]})
    test = pd.DataFrame({"f1": [10], "f2": [11], "target": [1]})

    train_path = tmp_path / "train.csv"
    valid_path = tmp_path / "valid.csv"
    test_path = tmp_path / "test.csv"

    train.to_csv(train_path, index=False)
    valid.to_csv(valid_path, index=False)
    test.to_csv(test_path, index=False)

    loader = DatasetLoader(random_state=7)
    cfg = {
        "source": {
            "type": "csv",
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "test_path": str(test_path),
            "target": "target",
        },
    }
    ds = loader.load(cfg)

    assert ds.metadata["strategy"] == "explicit_files"
    assert ds.X_train.shape == (3, 2)
    assert ds.X_valid.shape == (2, 2)
    assert ds.X_test.shape == (1, 2)
    assert ds.y_train.tolist() == [0, 1, 0]
