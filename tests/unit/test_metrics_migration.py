import numpy as np

from mlproj.evaluation.metrics import classification_metrics, regression_metrics


def test_classification_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    m = classification_metrics(y_true, y_pred)
    assert "accuracy" in m
    assert "f1" in m


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    m = regression_metrics(y_true, y_pred)
    assert "rmse" in m
    assert m["rmse"] >= 0
