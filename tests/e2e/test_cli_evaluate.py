import json

import pandas as pd

from mlproj.cli import main
from mlproj.training.trainer import Trainer


def test_cli_evaluate_with_model_uri(tmp_path):
    cfg = {
        "task": "classification",
        "artifact_root": str(tmp_path / "artifacts"),
        "source": {"type": "sklearn", "name": "wine"},
        "split": {"strategy": "random", "valid_size": 0.2, "test_size": 0.2},
        "model": {"name": "logistic_regression", "params": {}},
        "tune": {"enabled": False},
    }
    artifact = Trainer(artifact_root=cfg["artifact_root"]).train(cfg)

    eval_df = pd.read_csv("dataset/classification/test.csv")
    eval_path = tmp_path / "eval.csv"
    eval_df.to_csv(eval_path, index=False)

    metrics_path = tmp_path / "eval_metrics.json"
    code = main(
        [
            "evaluate",
            "--model-uri",
            str(artifact.model_uri),
            "--input",
            str(eval_path),
            "--target-col",
            "target",
            "--task",
            "classification",
            "--output-metrics",
            str(metrics_path),
        ]
    )
    assert code == 0
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["task"] == "classification"
    assert "accuracy" in metrics["metrics"]
