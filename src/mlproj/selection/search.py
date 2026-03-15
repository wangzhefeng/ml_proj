from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
)


@dataclass
class SearchResult:
    best_params: dict[str, Any]
    best_score: float
    best_estimator: Any


@dataclass
class NestedCVResult:
    outer_scores: list[float]
    mean_score: float
    std_score: float
    best_params_per_fold: list[dict[str, Any]]


def run_grid_search(
    estimator, X, y, param_grid: dict, cv: int = 5, scoring: str | None = None
) -> SearchResult:
    search = GridSearchCV(
        estimator=estimator, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=1
    )
    search.fit(X, y)
    return SearchResult(
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        best_estimator=search.best_estimator_,
    )


def run_halving_grid_search(
    estimator,
    X,
    y,
    param_grid: dict,
    cv: int = 5,
    scoring: str | None = None,
    factor: int = 3,
    random_state: int = 42,
) -> SearchResult:
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingGridSearchCV

    search = HalvingGridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        factor=factor,
        n_jobs=1,
        random_state=random_state,
    )
    search.fit(X, y)
    return SearchResult(
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        best_estimator=search.best_estimator_,
    )


def run_random_search(
    estimator,
    X,
    y,
    param_distributions: dict,
    n_iter: int = 10,
    cv: int = 5,
    scoring: str | None = None,
    random_state: int = 42,
) -> SearchResult:
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        random_state=random_state,
    )
    search.fit(X, y)
    return SearchResult(
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        best_estimator=search.best_estimator_,
    )


def run_halving_random_search(
    estimator,
    X,
    y,
    param_distributions: dict,
    cv: int = 5,
    scoring: str | None = None,
    factor: int = 3,
    n_candidates: str | int = "exhaust",
    random_state: int = 42,
) -> SearchResult:
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV

    search = HalvingRandomSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        cv=cv,
        scoring=scoring,
        factor=factor,
        n_candidates=n_candidates,
        n_jobs=1,
        random_state=random_state,
    )
    search.fit(X, y)
    return SearchResult(
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        best_estimator=search.best_estimator_,
    )


def run_nested_cv(
    estimator,
    X,
    y,
    param_grid: dict,
    inner_cv: int = 3,
    outer_cv: int = 5,
    scoring: str | None = None,
    task: str = "classification",
    random_state: int = 42,
) -> NestedCVResult:
    if task == "classification":
        outer_splitter = StratifiedKFold(
            n_splits=outer_cv,
            shuffle=True,
            random_state=random_state,
        )
        inner_splitter = StratifiedKFold(
            n_splits=inner_cv,
            shuffle=True,
            random_state=random_state,
        )
    else:
        outer_splitter = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
        inner_splitter = KFold(n_splits=inner_cv, shuffle=True, random_state=random_state)

    scorer = check_scoring(estimator, scoring=scoring)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    outer_scores: list[float] = []
    params_per_fold: list[dict[str, Any]] = []

    for train_idx, test_idx in outer_splitter.split(X_arr, y_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            cv=inner_splitter,
            scoring=scoring,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        score = scorer(search.best_estimator_, X_test, y_test)
        outer_scores.append(float(score))
        params_per_fold.append(search.best_params_)

    return NestedCVResult(
        outer_scores=outer_scores,
        mean_score=float(np.mean(outer_scores)),
        std_score=float(np.std(outer_scores)),
        best_params_per_fold=params_per_fold,
    )


def run_bayes_search_demo(
    func, pbounds: dict, init_points: int = 2, n_iter: int = 3
) -> dict[str, float]:
    try:
        from bayes_opt import BayesianOptimization
    except Exception as exc:
        raise RuntimeError("bayes_opt is not installed") from exc

    optimizer = BayesianOptimization(f=func, pbounds=pbounds, verbose=0, random_state=1)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer.max
