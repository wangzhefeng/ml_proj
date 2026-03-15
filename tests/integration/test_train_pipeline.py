import json
from pathlib import Path

from mlproj.training.trainer import Trainer


def test_train_classification_end_to_end(tmp_path):
    cfg = {
        "task": "classification",
        "artifact_root": str(tmp_path / "artifacts"),
        "source": {"type": "csv", "path": "dataset/classification/train.csv", "target": "target"},
        "split": {"strategy": "random", "valid_size": 0.2, "test_size": 0.2},
        "model": {"name": "logistic_regression", "params": {}},
        "tune": {"enabled": False},
        "random_state": 123,
    }

    artifact = Trainer(artifact_root=cfg["artifact_root"]).train(cfg)
    metrics = json.loads(artifact.metrics_uri.read_text(encoding="utf-8"))
    summary = json.loads(
        (Path(artifact.model_uri).parent / "summary.json").read_text(encoding="utf-8")
    )

    assert artifact.model_uri.exists()
    assert "accuracy" in metrics
    assert "run_metadata" in summary
    assert summary["run_metadata"]["random_seed"] == 123
    assert "git_commit" in summary["run_metadata"]
    assert "environment" in summary["run_metadata"]
