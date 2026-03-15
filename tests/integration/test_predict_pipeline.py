import pandas as pd
from sklearn.datasets import load_wine

from mlproj.inference.predictor import Predictor
from mlproj.training.trainer import Trainer


def test_predict_offline_file(tmp_path):
    cfg = {
        "task": "classification",
        "artifact_root": str(tmp_path / "artifacts"),
        "source": {"type": "sklearn", "name": "wine"},
        "split": {"strategy": "random", "valid_size": 0.2, "test_size": 0.2},
        "model": {"name": "logistic_regression", "params": {}},
        "tune": {"enabled": False},
    }
    artifact = Trainer(artifact_root=cfg["artifact_root"]).train(cfg)

    X = load_wine(as_frame=True).data.head(8)
    inp = tmp_path / "input.csv"
    out = tmp_path / "pred.csv"
    X.to_csv(inp, index=False)

    predictor = Predictor(artifact.model_uri)
    predictor.predict_file(inp, out)

    result = pd.read_csv(out)
    assert "prediction" in result.columns
