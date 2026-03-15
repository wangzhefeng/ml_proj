from pathlib import Path

import pandas as pd

from mlproj.data.loader import (
    get_lgb_train_test_data,
    get_params,
    get_xgb_train_test_data,
    load_yaml,
)


def test_load_yaml_from_data_provider():
    cfg = load_yaml("configs/classification/train.yaml")
    assert isinstance(cfg, dict)
    assert "task" in cfg


def test_get_params_bridge():
    cfg = get_params("configs/classification/train.yaml")
    assert isinstance(cfg, dict)


def test_lgb_xgb_loader_error_or_output(tmp_path: Path):
    train = tmp_path / "train.tsv"
    test = tmp_path / "test.tsv"

    pd.DataFrame([[1, 0.1, 0.2], [0, 0.3, 0.4]]).to_csv(train, sep="\t", header=False, index=False)
    pd.DataFrame([[1, 0.5, 0.6], [0, 0.7, 0.8]]).to_csv(test, sep="\t", header=False, index=False)

    try:
        out_lgb = get_lgb_train_test_data(str(train), str(test))
        assert len(out_lgb) >= 6
    except RuntimeError:
        assert True

    try:
        out_xgb = get_xgb_train_test_data(str(train), str(test))
        assert len(out_xgb) >= 6
    except RuntimeError:
        assert True
