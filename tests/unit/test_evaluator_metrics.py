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


def test_classification_multiclass_scores_are_supported():
    evaluator = Evaluator()
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    y_score = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.10, 0.20, 0.70],
            [0.80, 0.10, 0.10],
            [0.10, 0.35, 0.55],
            [0.05, 0.10, 0.85],
        ]
    )

    report = evaluator.evaluate(
        y_true=y_true, y_pred=y_pred, y_score=y_score, task="classification"
    )

    assert "auc" in report.metrics
    assert "pr_auc" in report.metrics


def test_clustering_metrics_allow_unsupervised_mode():
    evaluator = Evaluator()
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ]
    )
    y_pred = np.array([0, 0, 1, 1])

    report = evaluator.evaluate(
        y_true=None,
        y_pred=y_pred,
        task="clustering",
        X_for_cluster=X,
    )

    assert "n_samples" in report.metrics
    assert "n_clusters_pred" in report.metrics
    assert "silhouette" in report.metrics
    assert "davies_bouldin" in report.metrics
    assert "adjusted_rand" not in report.metrics
