from __future__ import annotations

from pathlib import Path

from mlproj.deploy import (
    create_iris_fastapi_app,
    create_iris_flask_app,
    export_linear_regression_onnx,
    export_resnet50_onnx,
    train_iris_classifier_model,
)


def _default_model_path() -> Path:
    return Path("artifacts") / "deploy" / "IrisClassifier.pkl"


def run_deploy_legacy_demo(script_path: str) -> dict[str, object]:
    normalized = script_path.replace("\\", "/").lower()
    name = Path(normalized).name

    model_path = _default_model_path()
    if not model_path.exists() and "model_training.py" not in normalized:
        train_iris_classifier_model(model_path)

    if name == "model_training.py":
        out = train_iris_classifier_model(model_path)
        return {"script": script_path, "method": "train_iris_model", **out}

    if name == "main.py" and "deploy_fastapi" in normalized:
        app = create_iris_fastapi_app(model_path)
        routes = sorted({route.path for route in app.router.routes})
        return {
            "script": script_path,
            "method": "fastapi_app",
            "model_uri": str(model_path),
            "routes": routes,
        }

    if name == "app.py" and "deploy_flask" in normalized:
        try:
            app = create_iris_flask_app(model_path)
            rules = sorted(rule.rule for rule in app.url_map.iter_rules())
            return {
                "script": script_path,
                "method": "flask_app",
                "model_uri": str(model_path),
                "routes": rules,
            }
        except RuntimeError as exc:
            return {
                "script": script_path,
                "method": "flask_app",
                "status": "skipped",
                "reason": str(exc),
            }

    if name == "main.py" and "deploy_flask" in normalized:
        try:
            app = create_iris_flask_app(model_path)
            client = app.test_client()
            resp = client.post("/predict", json={"request": "5.1,3.5,1.4,0.2"})
            return {
                "script": script_path,
                "method": "flask_predict_demo",
                "status_code": int(resp.status_code),
                "response": resp.get_json(),
            }
        except RuntimeError as exc:
            return {
                "script": script_path,
                "method": "flask_predict_demo",
                "status": "skipped",
                "reason": str(exc),
            }

    if name == "onnx_lr.py":
        out = export_linear_regression_onnx(
            Path("artifacts") / "deploy" / "onnx" / "linear_regression.onnx"
        )
        return {"script": script_path, "method": "onnx_lr", **out}

    if name == "onnx_resnet50.py":
        out = export_resnet50_onnx(Path("artifacts") / "deploy" / "onnx")
        return {"script": script_path, "method": "onnx_resnet50", **out}

    raise ValueError(f"Unsupported deploy legacy script path: {script_path}")
