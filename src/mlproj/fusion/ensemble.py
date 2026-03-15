from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
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
