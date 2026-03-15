from __future__ import annotations

from typing import Any

import numpy as np


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
