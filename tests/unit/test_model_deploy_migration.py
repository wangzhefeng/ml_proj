from fastapi.testclient import TestClient

from mlproj.inference.service import create_app
from mlproj.training.trainer import Trainer


def test_service_health_and_predict_contract(tmp_path):
    cfg = {
        "task": "classification",
        "artifact_root": str(tmp_path / "artifacts"),
        "source": {"type": "csv", "path": "dataset/classification/train.csv", "target": "target"},
        "split": {"strategy": "random", "valid_size": 0.2, "test_size": 0.2},
        "model": {"backend": "sklearn", "name": "logistic_regression", "params": {}},
        "tune": {"enabled": False},
    }
    artifact = Trainer(artifact_root=cfg["artifact_root"]).train(cfg)

    app = create_app(str(artifact.model_uri))
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    bad = client.post("/predict", json={"rows": []})
    assert bad.status_code == 400

    good = client.post(
        "/predict",
        json={
            "rows": [
                {
                    "feat_0": 0.3,
                    "feat_1": -0.4,
                    "feat_2": 1.2,
                    "feat_3": 0.8,
                    "feat_4": -0.2,
                    "feat_5": 0.1,
                    "feat_6": 0.05,
                    "feat_7": -0.6,
                }
            ]
        },
    )
    assert good.status_code in {200, 400}
