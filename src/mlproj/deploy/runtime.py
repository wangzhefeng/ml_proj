from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_iris_classifier_model(model_path: str | Path) -> dict[str, Any]:
    df = pd.read_csv("dataset/classification/train.csv")
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    pipe = Pipeline([("clf", LogisticRegression(max_iter=1000, multi_class="auto"))])
    pipe.fit(X, y)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    return {"model_uri": str(model_path), "rows": int(X.shape[0]), "features": int(X.shape[1])}


def parse_feature_request(params_str: str) -> list[float]:
    return [float(item) for item in params_str.split(",")]


def _load_model(model_path: str | Path):
    return joblib.load(Path(model_path))


def create_iris_fastapi_app(model_path: str | Path) -> FastAPI:
    app = FastAPI(
        title="Classification Prediction Model API",
        description="Predict class with trained sklearn model",
        version="0.1",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/predict-result")
    def predict_result(request: str) -> dict[str, str]:
        model = _load_model(model_path)
        values = parse_feature_request(request)
        prediction = int(model.predict([values])[0])
        prob = float(model.predict_proba([values])[0, prediction])
        return {"prediction": str(prediction), "probability": f"{prob:.2f}"}

    return app


def create_iris_flask_app(model_path: str | Path):
    try:
        from flask import Flask, jsonify, request
    except Exception as exc:
        raise RuntimeError("flask is not installed") from exc

    app = Flask(__name__)

    @app.get("/")
    def root():
        return jsonify({"msg": "POST /predict with JSON payload: {'request': 'f1,f2,...'}"})

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        req = payload.get("request", "")
        if not req:
            return jsonify({"error": "request is required"}), 400

        model = _load_model(model_path)
        values = parse_feature_request(req)
        prediction = int(model.predict([values])[0])
        prob = float(model.predict_proba([values])[0, prediction])
        return jsonify({"prediction": str(prediction), "probability": f"{prob:.2f}"})

    return app


def export_linear_regression_onnx(output_path: str | Path) -> dict[str, Any]:
    try:
        import onnx
        from onnx import TensorProto
        from onnx.helper import (
            make_graph,
            make_model,
            make_node,
            make_tensor_value_info,
        )
    except Exception:
        return {"status": "skipped", "reason": "onnx dependency not installed"}

    X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

    node1 = make_node("MatMul", ["X", "A"], ["XA"])
    node2 = make_node("Add", ["XA", "B"], ["Y"])
    graph = make_graph([node1, node2], "lr", [X, A, B], [Y])
    model = make_model(graph)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(model.SerializeToString())

    return {
        "status": "ok",
        "onnx_version": getattr(onnx, "__version__", "unknown"),
        "output": str(output_path),
    }


def export_resnet50_onnx(output_dir: str | Path) -> dict[str, Any]:
    try:
        import torch
        import torchvision
    except Exception:
        return {"status": "skipped", "reason": "torch/torchvision dependency not installed"}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = torchvision.models.resnet50(weights=None)
    model.eval()

    dummy_data = torch.randn((1, 3, 224, 224))
    dynamic_output = output_dir / "resnet50_bs_dynamic.onnx"

    torch.onnx.export(
        model,
        (dummy_data),
        str(dynamic_output),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_axes"}, "output": {0: "batch_axes"}},
    )

    return {
        "status": "ok",
        "output": str(dynamic_output),
        "note": "weights=None to avoid network download in migration bridge",
    }
