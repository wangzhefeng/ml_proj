from __future__ import annotations

import itertools
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
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
