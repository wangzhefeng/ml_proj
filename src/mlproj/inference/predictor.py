from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


class Predictor:
    def __init__(self, model_uri: str | Path) -> None:
        self.model_uri = Path(model_uri)
        self.bundle = joblib.load(self.model_uri)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        pre = self.bundle["preprocessor"]
        feats = self.bundle["features"]
        est = self.bundle["estimator"]

        X = feats.transform(pre.transform(df))
        preds = est.predict(X)
        out = df.copy()
        out["prediction"] = preds

        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X)
            if getattr(proba, "ndim", 1) == 2:
                out["prediction_score"] = proba.max(axis=1)
        return out

    def predict_file(self, input_path: str | Path, output_path: str | Path) -> Path:
        input_path = Path(input_path)
        output_path = Path(output_path)

        if input_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)

        result = self.predict_dataframe(df)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".parquet":
            result.to_parquet(output_path, index=False)
        else:
            result.to_csv(output_path, index=False)

        report = {"rows": len(result), "output": str(output_path)}
        output_path.with_suffix(".report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return output_path
