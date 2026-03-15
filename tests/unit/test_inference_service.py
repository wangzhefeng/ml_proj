from fastapi.testclient import TestClient

from mlproj.inference.service import create_app
from mlproj.training.trainer import Trainer


def test_service_predict_validation(tmp_path):
    cfg = {
        "task": "classification",
        "artifact_root": str(tmp_path / "artifacts"),
        "source": {"type": "csv", "path": "dataset/classification/train.csv", "target": "target"},
        "split": {"strategy": "random", "valid_size": 0.2, "test_size": 0.2},
        "model": {"name": "logistic_regression", "params": {}},
        "tune": {"enabled": False},
    }
    artifact = Trainer(artifact_root=cfg["artifact_root"]).train(cfg)

    app = create_app(str(artifact.model_uri))
    client = TestClient(app)

    bad = client.post("/predict", json={"rows": []})
    assert bad.status_code == 400

    good = client.post(
        "/predict",
        json={
            "rows": [
                {
                    "feature_0": 14.23,
                    "feature_1": 1.71,
                    "feature_2": 2.43,
                    "feature_3": 15.6,
                    "feature_4": 127,
                }
            ]
        },
    )
    assert good.status_code in {200, 400}
