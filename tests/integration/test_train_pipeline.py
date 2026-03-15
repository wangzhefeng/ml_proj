import json

from mlproj.training.trainer import Trainer


def test_train_classification_end_to_end(tmp_path):
    cfg = {
        "task": "classification",
        "artifact_root": str(tmp_path / "artifacts"),
        "source": {"type": "sklearn", "name": "wine"},
        "split": {"strategy": "random", "valid_size": 0.2, "test_size": 0.2},
        "model": {"name": "logistic_regression", "params": {}},
        "tune": {"enabled": False},
    }

    artifact = Trainer(artifact_root=cfg["artifact_root"]).train(cfg)
    metrics = json.loads(artifact.metrics_uri.read_text(encoding="utf-8"))

    assert artifact.model_uri.exists()
    assert "accuracy" in metrics
