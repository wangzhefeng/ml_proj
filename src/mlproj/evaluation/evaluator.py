from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)

from mlproj.types import MetricReport


class Evaluator:
    def evaluate(
        self,
        y_true,
        y_pred,
        y_score=None,
        task: str = "classification",
        X_for_cluster=None,
    ) -> MetricReport:
        if task == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, average="weighted")),
            }
            if y_score is not None:
                try:
                    if np.ndim(y_score) == 2 and y_score.shape[1] > 1:
                        metrics["auc"] = float(roc_auc_score(y_true, y_score, multi_class="ovr"))
                    else:
                        metrics["auc"] = float(roc_auc_score(y_true, y_score))
                except Exception:
                    pass
            return MetricReport(task=task, metrics=metrics)

        if task in {"regression", "timeseries"}:
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics = {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": rmse,
                "r2": float(r2_score(y_true, y_pred)),
            }
            return MetricReport(task=task, metrics=metrics)

        if task == "clustering":
            if X_for_cluster is None:
                raise ValueError("X_for_cluster is required for clustering metrics")
            metrics = {
                "silhouette": float(silhouette_score(X_for_cluster, y_pred)),
                "davies_bouldin": float(davies_bouldin_score(X_for_cluster, y_pred)),
            }
            return MetricReport(task=task, metrics=metrics)

        raise ValueError(f"Unsupported task: {task}")
