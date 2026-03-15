from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI

from mlproj.inference.predictor import Predictor


def create_app(model_uri: str) -> FastAPI:
    predictor = Predictor(model_uri=model_uri)
    app = FastAPI(title="mlproj inference", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(payload: dict[str, Any]) -> dict[str, Any]:
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            return {"error": "rows must be a list of objects"}
        df = pd.DataFrame(rows)
        result = predictor.predict_dataframe(df)
        return {"rows": result.to_dict(orient="records")}

    return app
