import numpy as np
import pandas as pd

from mlproj.features import binarization
from mlproj.evaluation.evaluator import precision_recall_curve_report


def test_feature_engine_dispatch_bridge():
    series = pd.Series([0.1, 0.4, 0.8, 0.2])
    out = binarization(series, threshold=0.3)
    assert out.shape[0] == len(series)


def test_metric_dispatch_bridge():
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.1, 0.8, 0.7, 0.2])
    rep = precision_recall_curve_report(y_true, y_score)
    assert "average_precision" in rep
