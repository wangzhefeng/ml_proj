from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mlproj.fusion import (
    average_predictions,
    build_adaboost_classifier,
    build_bagging_classifier,
    build_stacking_classifier,
    build_voting_classifier,
    rank_models_by_score,
)


def run_fusion_legacy_demo(script_path: str) -> dict[str, object]:
    name = Path(script_path.replace("\\", "/").lower()).name
    X, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    if name == "voting.py":
        model = build_voting_classifier(voting="hard")
        model.fit(x_train, y_train)
        score = float(model.score(x_test, y_test))
        return {"script": script_path, "method": "voting", "score": score}

    if name == "stacking.py":
        model = build_stacking_classifier()
        model.fit(x_train, y_train)
        score = float(model.score(x_test, y_test))
        return {"script": script_path, "method": "stacking", "score": score}

    if name == "baggging.py":
        model = build_bagging_classifier(n_estimators=6)
        model.fit(x_train, y_train)
        score = float(model.score(x_test, y_test))
        return {"script": script_path, "method": "bagging", "score": score}

    if name == "boosting.py":
        model = build_adaboost_classifier()
        model.fit(x_train, y_train)
        score = float(model.score(x_test, y_test))
        return {"script": script_path, "method": "boosting", "score": score}

    if name == "averaging.py":
        sample_preds = [
            np.array([0.1, 0.4, 0.8]),
            np.array([0.2, 0.6, 0.7]),
            np.array([0.0, 0.5, 0.9]),
        ]
        avg = average_predictions(sample_preds)
        return {"script": script_path, "method": "averaging", "avg": avg.tolist()}

    if name == "sorting.py":
        scores = {"model_a": 0.82, "model_b": 0.76, "model_c": 0.91}
        ranked = rank_models_by_score(scores)
        return {"script": script_path, "method": "sorting", "ranked": ranked}

    if name == "pystacknet.py":
        # Optional dependency fallback: use sklearn stacking as runtime-compatible replacement.
        model = build_stacking_classifier()
        model.fit(x_train, y_train)
        score = float(model.score(x_test, y_test))
        return {
            "script": script_path,
            "method": "pystacknet_compat",
            "score": score,
            "note": "pystacknet optional; fallback to sklearn stacking in framework v1",
        }

    raise ValueError(f"Unsupported fusion legacy script path: {script_path}")
