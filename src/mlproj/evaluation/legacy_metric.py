from __future__ import annotations

import itertools
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


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
