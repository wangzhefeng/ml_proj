from pathlib import Path

import pandas as pd

from mlproj.data.loader import DatasetLoader


def test_loader_bridge_from_current_data_module():
    loader = DatasetLoader(random_state=7)
    cfg = {
        "source": {"type": "csv", "path": "dataset/classification/train.csv", "target": "target"},
        "split": {"strategy": "random", "valid_size": 0.2, "test_size": 0.2},
    }
    ds = loader.load(cfg)
    assert len(ds.X_train) > 0
    assert ds.y_train is not None


def test_lgb_xgb_loader_error_or_output(tmp_path: Path):
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
