from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlproj.inference.predictor import Predictor


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)


class PredictResponse(BaseModel):
    rows: list[dict[str, Any]]


def create_app(model_uri: str) -> FastAPI:
    predictor = Predictor(model_uri=model_uri)
    app = FastAPI(title="mlproj inference", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        if not payload.rows:
            raise HTTPException(status_code=400, detail="rows must be a non-empty list of objects")
        try:
            df = pd.DataFrame(payload.rows)
            result = predictor.predict_dataframe(df)
            return PredictResponse(rows=result.to_dict(orient="records"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
