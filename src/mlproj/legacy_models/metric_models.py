from __future__ import annotations

from pathlib import Path

import numpy as np

from mlproj.evaluation.report import precision_recall_curve_report


def run_metric_legacy_demo(script_path: str) -> dict[str, object]:
    name = Path(script_path.replace("\\", "/").lower()).name
    if name == "metric_report.py":
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_score = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.4])
        rep = precision_recall_curve_report(y_true, y_score)
        return {"script": script_path, "method": "metric_report", "ap": rep["average_precision"]}
    return {"script": script_path, "method": "metric_bridge"}
