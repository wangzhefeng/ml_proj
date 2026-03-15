from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class Predictor:
    def __init__(self, model_uri: str | Path) -> None:
        self.model_uri = Path(model_uri)
        self.bundle = joblib.load(self.model_uri)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        pre = self.bundle.get("preprocessor")
        feats = self.bundle.get("features")
        est = self.bundle["estimator"]
        task = self.bundle.get("task")

        X = df.copy()
        if pre is not None:
            X = pre.transform(X)
        if feats is not None:
            X = feats.transform(X)

        task = task or self._infer_task(est)

        if task == "pca_reduction":
            transformed = est.transform(X)
            out = df.copy()
            transformed_arr = np.asarray(transformed)
            n_components = transformed_arr.shape[1] if transformed_arr.ndim == 2 else 1
            if transformed_arr.ndim == 1:
                transformed_arr = transformed_arr.reshape(-1, 1)
            for idx in range(n_components):
                out[f"pca_{idx}"] = transformed_arr[:, idx]
            return out

        if task == "topic_modeling":
            topic_dist = est.transform(X)
            out = df.copy()
            topic_arr = np.asarray(topic_dist)
            for idx in range(topic_arr.shape[1]):
                out[f"topic_{idx}"] = topic_arr[:, idx]
            out["topic_prediction"] = np.argmax(topic_arr, axis=1)
            return out

        out = df.copy()
        preds = est.predict(X)
        out["prediction"] = preds

        if task == "anomaly_detection":
            if set(np.unique(np.asarray(preds)).tolist()).issubset({-1, 1}):
                out["anomaly_label"] = (np.asarray(preds) == -1).astype(int)
            else:
                out["anomaly_label"] = (np.asarray(preds).astype(float) > 0).astype(int)
            if hasattr(est, "score_samples"):
                out["anomaly_score"] = est.score_samples(X)
            return out

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
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return output_path

    def _infer_task(self, estimator) -> str:
        if hasattr(estimator, "vectorizer") and hasattr(estimator, "model"):
            return "topic_modeling"
        if hasattr(estimator, "explained_variance_ratio_"):
            return "pca_reduction"
        if hasattr(estimator, "score_samples") and not hasattr(estimator, "predict_proba"):
            return "anomaly_detection"
        if hasattr(estimator, "cluster_centers_"):
            return "clustering"
        return "classification"
