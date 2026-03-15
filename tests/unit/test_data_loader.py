import pandas as pd
import pytest

from mlproj.data.loader import DatasetLoader


def test_loader_csv_random_split():
    loader = DatasetLoader(random_state=7)
    cfg = {
        "source": {"type": "csv", "path": "dataset/classification/train.csv", "target": "target"},
        "split": {"strategy": "random", "valid_size": 0.2, "test_size": 0.2},
    }
    ds = loader.load(cfg)
    assert len(ds.X_train) > 0
    assert ds.y_train is not None
    assert len(ds.X_train.columns) > 0


def test_loader_rejects_non_csv_source():
    loader = DatasetLoader(random_state=7)
    with pytest.raises(ValueError):
        loader.load({"source": {"type": "sklearn", "name": "wine"}})


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


def test_loader_lgb_xgb_error_or_output(tmp_path):
    train = tmp_path / "train.tsv"
    test = tmp_path / "test.tsv"

    pd.DataFrame([[1, 0.1, 0.2], [0, 0.3, 0.4]]).to_csv(train, sep="\t", header=False, index=False)
    pd.DataFrame([[1, 0.5, 0.6], [0, 0.7, 0.8]]).to_csv(test, sep="\t", header=False, index=False)

    loader = DatasetLoader(random_state=7)

    try:
        out_lgb = loader.load_lgb_train_test_data(str(train), str(test))
        assert len(out_lgb) >= 6
    except RuntimeError:
        assert True

    try:
        out_xgb = loader.load_xgb_train_test_data(str(train), str(test))
        assert len(out_xgb) >= 6
    except RuntimeError:
        assert True


def test_loader_uses_boosting_branch_for_xgb(monkeypatch):
    loader = DatasetLoader(random_state=7)
    X_train = pd.DataFrame(
        {"f1": [0.1, 0.2, 0.3, 0.4, 0.45, 0.55], "f2": [1.0, 1.1, 1.2, 1.3, 1.35, 1.45]}
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1])
    X_test = pd.DataFrame({"f1": [0.5, 0.6], "f2": [1.4, 1.5]})
    y_test = pd.Series([1, 0])

    def _fake_load_xgb(self, train_path, test_path, weight_paths=None):
        assert train_path == "train.tsv"
        assert test_path == "test.tsv"
        return X_train, y_train, X_test, y_test, "dtrain_obj", "dtest_obj"

    monkeypatch.setattr(DatasetLoader, "load_xgb_train_test_data", _fake_load_xgb)

    ds = loader.load(
        {
            "model": {"name": "xgboost"},
            "source": {"type": "csv", "train_path": "train.tsv", "test_path": "test.tsv"},
            "split": {"valid_size": 0.34},
        }
    )

    assert ds.metadata["backend"] == "xgboost"
    assert ds.metadata["backend_train"] == "dtrain_obj"
    assert ds.metadata["backend_test"] == "dtest_obj"
    assert ds.metadata["strategy"] == "boosting_explicit_train_test"
    assert ds.X_train.shape[1] == 2
    assert ds.X_valid is not None
    assert ds.X_test.shape == (2, 2)
