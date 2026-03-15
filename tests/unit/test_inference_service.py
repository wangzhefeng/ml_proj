from fastapi.testclient import TestClient

from mlproj.inference.service import create_app
from mlproj.training.trainer import Trainer


def test_service_predict_validation(tmp_path):
    cfg = {
        "task": "classification",
        "artifact_root": str(tmp_path / "artifacts"),
        "source": {"type": "sklearn", "name": "wine"},
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
                    "alcohol": 14.23,
                    "malic_acid": 1.71,
                    "ash": 2.43,
                    "alcalinity_of_ash": 15.6,
                    "magnesium": 127,
                    "total_phenols": 2.8,
                    "flavanoids": 3.06,
                    "nonflavanoid_phenols": 0.28,
                    "proanthocyanins": 2.29,
                    "color_intensity": 5.64,
                    "hue": 1.04,
                    "od280/od315_of_diluted_wines": 3.92,
                    "proline": 1065,
                }
            ]
        },
    )
    assert good.status_code == 200
    body = good.json()
    assert "rows" in body
    assert len(body["rows"]) == 1
