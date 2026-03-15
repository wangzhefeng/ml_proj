from pathlib import Path

from mlproj.deploy import (
    create_iris_fastapi_app,
    export_linear_regression_onnx,
    export_resnet50_onnx,
    train_iris_classifier_model,
)


def test_train_and_fastapi_app(tmp_path):
    model_path = tmp_path / "IrisClassifier.pkl"
    out = train_iris_classifier_model(model_path)
    assert Path(out["model_uri"]).exists()

    app = create_iris_fastapi_app(model_path)
    route_paths = {route.path for route in app.router.routes}
    assert "/health" in route_paths
    assert "/predict-result" in route_paths


def test_onnx_export_bridges(tmp_path):
    lr_out = export_linear_regression_onnx(tmp_path / "linear_regression.onnx")
    assert lr_out["status"] in {"ok", "skipped"}

    resnet_out = export_resnet50_onnx(tmp_path)
    assert resnet_out["status"] in {"ok", "skipped"}
