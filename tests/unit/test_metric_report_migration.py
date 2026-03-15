import numpy as np

from mlproj.evaluation.report import (
    ROC_plot,
    confusion_matrix_report,
    precision_recall_curve_report,
)


def test_metric_report_bridges():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_score = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.4])
    y_pred = (y_score >= 0.5).astype(int)

    pr = precision_recall_curve_report(y_true, y_score)
    assert "average_precision" in pr

    roc = ROC_plot(y_true, y_score)
    assert "auc" in roc

    cm = confusion_matrix_report(y_true, y_pred, classes=["neg", "pos"], normalize=True)
    assert cm.shape == (2, 2)
