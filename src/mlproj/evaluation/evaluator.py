from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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
                        metrics["pr_auc"] = float(
                            average_precision_score(y_true, y_score, average="weighted")
                        )
                    else:
                        metrics["auc"] = float(roc_auc_score(y_true, y_score))
                        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
                except Exception:
                    pass
            return MetricReport(task=task, metrics=metrics)

        if task in {"regression", "timeseries"}:
            metrics = self._regression_metrics(y_true, y_pred)
            if task == "timeseries":
                metrics.update(self._timeseries_window_metrics(y_true, y_pred))
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

    def _regression_metrics(self, y_true, y_pred) -> dict[str, float]:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)

        rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
        r2 = float(r2_score(y_true_arr, y_pred_arr))
        mape = self._safe_mape(y_true_arr, y_pred_arr)
        smape = self._safe_smape(y_true_arr, y_pred_arr)

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "smape": smape,
        }

    def _timeseries_window_metrics(self, y_true, y_pred) -> dict[str, float]:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)

        n = len(y_true_arr)
        if n < 4:
            return {
                "rolling_window": float(n),
                "rolling_rmse_mean": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
                "rolling_rmse_std": 0.0,
            }

        window = max(3, min(20, n // 5))
        rmses: list[float] = []
        for end in range(window, n + 1):
            s = end - window
            rmses.append(float(np.sqrt(mean_squared_error(y_true_arr[s:end], y_pred_arr[s:end]))))

        return {
            "rolling_window": float(window),
            "rolling_rmse_mean": float(np.mean(rmses)),
            "rolling_rmse_std": float(np.std(rmses)),
        }

    def _safe_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        denom = np.clip(np.abs(y_true), 1e-8, None)
        return float(np.mean(np.abs((y_true - y_pred) / denom)))

    def _safe_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        denom = np.clip(np.abs(y_true) + np.abs(y_pred), 1e-8, None)
        return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))
