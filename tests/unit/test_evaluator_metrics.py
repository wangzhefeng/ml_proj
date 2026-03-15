import numpy as np

from mlproj.evaluation.evaluator import Evaluator


def test_classification_metrics_include_pr_auc():
    evaluator = Evaluator()
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0])
    y_score = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3])

    report = evaluator.evaluate(
        y_true=y_true, y_pred=y_pred, y_score=y_score, task="classification"
    )

    assert "pr_auc" in report.metrics
    assert "auc" in report.metrics


def test_regression_metrics_include_mape_smape():
    evaluator = Evaluator()
    y_true = np.array([10.0, 12.0, 11.5, 13.0, 15.0])
    y_pred = np.array([9.8, 11.6, 11.0, 13.3, 14.7])

    report = evaluator.evaluate(y_true=y_true, y_pred=y_pred, task="regression")

    assert "mape" in report.metrics
    assert "smape" in report.metrics


def test_timeseries_metrics_include_rolling_window():
    evaluator = Evaluator()
    y_true = np.array([float(i) for i in range(1, 41)])
    y_pred = y_true + 0.5

    report = evaluator.evaluate(y_true=y_true, y_pred=y_pred, task="timeseries")

    assert "rolling_window" in report.metrics
    assert "rolling_rmse_mean" in report.metrics
    assert "rolling_rmse_std" in report.metrics
