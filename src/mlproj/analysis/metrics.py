from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


def classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_score is not None:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_score, multi_class="ovr"))
        except Exception:
            pass
    return out


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def clustering_metrics(X, labels_pred, labels_true=None) -> dict[str, float]:
    metrics = {
        "silhouette": float(silhouette_score(X, labels_pred)),
        "davies_bouldin": float(davies_bouldin_score(X, labels_pred)),
    }
    if labels_true is not None:
        metrics["adjusted_rand"] = float(adjusted_rand_score(labels_true, labels_pred))
    return metrics


@dataclass
class BinaryScore:
    def accuracy(self, y_true, y_pred) -> float:
        return classification_metrics(y_true, y_pred)["accuracy"]
