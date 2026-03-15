from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    average_precision_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.preprocessing import label_binarize

from mlproj.types import MetricReport

matplotlib.use("Agg")


def precision_recall_curve_report(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)

    plt.figure()
    plt.step(recall, precision, color="b", alpha=0.4, where="post")
    plt.fill_between(recall, precision, alpha=0.2, color="b", step="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f"Precision-Recall Curve: AP={average_precision:.3f}")

    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "average_precision": float(average_precision),
    }


def ROC_plot(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = float(np.trapezoid(tpr, fpr))

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label=f"ROC curve (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": roc_auc}


def confusion_matrix_report(
    y_true,
    y_pred,
    classes,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap=plt.cm.Blues,
):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    return cm


def param_cvsearch_report(search_instance, n_top: int = 3) -> list[dict[str, Any]]:
    results = search_instance.cv_results_
    report: list[dict[str, Any]] = []
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            report.append(
                {
                    "rank": int(i),
                    "mean_test_score": float(results["mean_test_score"][candidate]),
                    "std_test_score": float(results["std_test_score"][candidate]),
                    "params": results["params"][candidate],
                }
            )
    return report


def classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(
            precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
    }
    if y_score is not None:
        out.update(_score_based_classification_metrics(y_true_arr, y_score))
    return out


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mape = _safe_mape(y_true_arr, y_pred_arr)
    smape = _safe_smape(y_true_arr, y_pred_arr)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "smape": smape,
    }


def clustering_metrics(X, labels_pred, labels_true=None) -> dict[str, float]:
    x_arr = np.asarray(X)
    labels_pred_arr = np.asarray(labels_pred)
    unique_labels = np.unique(labels_pred_arr)

    metrics: dict[str, float] = {
        "n_samples": float(len(labels_pred_arr)),
        "n_clusters_pred": float(len(unique_labels)),
    }
    if len(unique_labels) >= 2 and len(labels_pred_arr) > len(unique_labels):
        metrics["silhouette"] = float(silhouette_score(x_arr, labels_pred_arr))
        metrics["davies_bouldin"] = float(davies_bouldin_score(x_arr, labels_pred_arr))
    else:
        metrics["silhouette"] = float("nan")
        metrics["davies_bouldin"] = float("nan")

    if labels_true is not None:
        metrics["adjusted_rand"] = float(adjusted_rand_score(labels_true, labels_pred_arr))
    return metrics


def _score_based_classification_metrics(y_true: np.ndarray, y_score: Any) -> dict[str, float]:
    score_arr = np.asarray(y_score)
    out: dict[str, float] = {}

    try:
        if score_arr.ndim == 2 and score_arr.shape[1] > 1:
            out["auc"] = float(roc_auc_score(y_true, score_arr, multi_class="ovr"))
            out["pr_auc"] = _multiclass_pr_auc(y_true, score_arr)
        else:
            score_1d = score_arr.reshape(-1)
            out["auc"] = float(roc_auc_score(y_true, score_1d))
            out["pr_auc"] = float(average_precision_score(y_true, score_1d))
    except Exception:
        return out
    return out


def _multiclass_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    classes = np.unique(y_true)
    if len(classes) <= 2:
        return float(average_precision_score(y_true, y_score[:, -1]))

    y_true_bin = label_binarize(y_true, classes=classes)
    usable_cols = min(y_true_bin.shape[1], y_score.shape[1])
    if usable_cols == 0:
        raise ValueError("Invalid y_score for multiclass PR-AUC")

    return float(
        average_precision_score(
            y_true_bin[:, :usable_cols],
            y_score[:, :usable_cols],
            average="macro",
        )
    )


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def _safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true) + np.abs(y_pred), 1e-8, None)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


@dataclass
class BinaryScore:
    def accuracy(self, y_true, y_pred) -> float:
        return classification_metrics(y_true, y_pred)["accuracy"]


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
            return MetricReport(task=task, metrics=classification_metrics(y_true, y_pred, y_score))

        if task == "regression":
            metrics = regression_metrics(y_true, y_pred)
            return MetricReport(task=task, metrics=metrics)

        if task == "clustering":
            if X_for_cluster is None:
                raise ValueError("X_for_cluster is required for clustering metrics")
            return MetricReport(
                task=task,
                metrics=clustering_metrics(X_for_cluster, y_pred, labels_true=y_true),
            )

        raise ValueError(f"Unsupported task: {task}")
