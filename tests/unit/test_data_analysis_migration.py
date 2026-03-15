from pathlib import Path

import pandas as pd

from mlproj.config import load_config
from mlproj.data.loader import DatasetLoader


def test_train_config_loads_for_current_architecture():
    cfg = load_config("configs/classification/train.yaml")
    assert cfg["task"] == "classification"
    assert cfg["model"]["name"] in {"logistic_regression", "random_forest"}


def test_evaluate_config_loads_for_current_architecture(tmp_path: Path):
    model_uri = tmp_path / "dummy.joblib"
    model_uri.write_bytes(b"x")
    input_csv = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(input_csv, index=False)

    eval_cfg = tmp_path / "eval.yaml"
    eval_cfg.write_text(
        "\n".join(
            [
                f"model_uri: {model_uri.as_posix()}",
                f"input: {input_csv.as_posix()}",
                "task: classification",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(eval_cfg)
    assert cfg["task"] == "classification"
    assert cfg["model_uri"].endswith("dummy.joblib")


def test_data_loader_supports_random_split_only():
    loader = DatasetLoader(random_state=7)
    cfg = {
        "source": {
            "type": "csv",
            "path": "dataset/regression/train.csv",
            "target": "target",
        },
        "split": {"strategy": "random", "valid_size": 0.2, "test_size": 0.2},
    }

    ds = loader.load(cfg)
    assert ds.metadata["strategy"] == "random"
    assert len(ds.X_train) > 0
    assert ds.X_valid is not None
    assert ds.X_test is not None
