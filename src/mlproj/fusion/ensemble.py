from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_voting_classifier(random_state: int = 42, voting: str = "hard") -> VotingClassifier:
    estimators = [
        ("dtc", DecisionTreeClassifier(random_state=random_state)),
        ("lr", LogisticRegression(max_iter=1000)),
        ("knn", KNeighborsClassifier()),
        ("svc", SVC(probability=(voting == "soft"), random_state=random_state)),
    ]
    return VotingClassifier(estimators=estimators, voting=voting)


def build_stacking_classifier(random_state: int = 42) -> StackingClassifier:
    base_learners = [
        ("knn", KNeighborsClassifier()),
        ("dt", DecisionTreeClassifier(random_state=random_state)),
        ("svc", SVC(probability=True, random_state=random_state)),
    ]
    return StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
    )


def build_bagging_classifier(random_state: int = 42, n_estimators: int = 10) -> BaggingClassifier:
    return BaggingClassifier(
        estimator=SVC(probability=True, random_state=random_state),
        n_estimators=n_estimators,
        random_state=random_state,
    )


def build_adaboost_classifier(random_state: int = 42) -> AdaBoostClassifier:
    base = DecisionTreeClassifier(max_depth=2, random_state=random_state)
    return AdaBoostClassifier(
        estimator=base,
        n_estimators=20,
        learning_rate=0.1,
        random_state=random_state,
    )


def average_predictions(predictions: Iterable[np.ndarray], axis: int = 0) -> np.ndarray:
    pred_list = [np.asarray(p) for p in predictions]
    if not pred_list:
        raise ValueError("predictions must not be empty")
    return np.mean(np.stack(pred_list, axis=0), axis=axis)


def rank_models_by_score(
    scores: dict[str, float], descending: bool = True
) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=descending)

def _load_classification_train_test(
    path: str = "dataset/classification/train.csv",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def run_fusion_legacy_demo(script_path: str) -> dict[str, object]:
    name = Path(script_path.replace("\\", "/").lower()).name

    if name in {"voting.py", "stacking.py", "baggging.py", "boosting.py", "pystacknet.py"}:
        x_train, x_test, y_train, y_test = _load_classification_train_test()

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
