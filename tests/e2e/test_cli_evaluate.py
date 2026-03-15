import json

import pandas as pd

from mlproj.cli import main
from mlproj.training.trainer import Trainer


def test_cli_evaluate_with_model_uri(tmp_path):
    cfg = {
        "task": "classification",
        "artifact_root": str(tmp_path / "artifacts"),
        "source": {"type": "csv", "path": "dataset/classification/train.csv", "target": "target"},
        "split": {"strategy": "random", "valid_size": 0.2, "test_size": 0.2},
        "model": {"backend": "sklearn", "name": "logistic_regression", "params": {}},
        "tune": {"enabled": False},
        "feature_pipeline": [],
    }
    artifact = Trainer(artifact_root=cfg["artifact_root"]).train(cfg)

    eval_df = pd.read_csv("dataset/classification/test.csv")
    eval_path = tmp_path / "eval.csv"
    eval_df.to_csv(eval_path, index=False)

    eval_cfg_path = tmp_path / "evaluate.yaml"
    eval_cfg_path.write_text(
        f"""model_uri: {artifact.model_uri.as_posix()}
input: {eval_path.as_posix()}
target_col: target
task: classification
output_metrics: {(tmp_path / 'eval_metrics.json').as_posix()}
""",
        encoding="utf-8",
    )

    code = main(["evaluate", "--config-yaml", str(eval_cfg_path)])
    assert code == 0

    metrics_path = tmp_path / "eval_metrics.json"
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["task"] == "classification"
    assert "accuracy" in metrics["metrics"]
