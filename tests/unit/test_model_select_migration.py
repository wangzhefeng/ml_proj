from scipy.stats import randint
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from mlproj.selection.search import (
    run_halving_grid_search,
    run_halving_random_search,
    run_nested_cv,
)


def test_halving_grid_search_bridge():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=180,
        n_features=10,
        n_informative=6,
        random_state=0,
    )
    res = run_halving_grid_search(
        estimator=RandomForestClassifier(random_state=0),
        X=X,
        y=y,
        param_grid={"max_depth": [2, None], "min_samples_split": [2, 4]},
        cv=3,
        scoring="accuracy",
        factor=2,
    )
    assert isinstance(res.best_params, dict)


def test_halving_random_search_bridge():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=180,
        n_features=10,
        n_informative=6,
        random_state=1,
    )
    res = run_halving_random_search(
        estimator=RandomForestClassifier(random_state=1),
        X=X,
        y=y,
        param_distributions={
            "max_depth": [2, None],
            "min_samples_split": randint(2, 6),
        },
        cv=3,
        scoring="accuracy",
        factor=2,
        n_candidates=6,
        random_state=1,
    )
    assert isinstance(res.best_params, dict)


def test_nested_cv_bridge_classification():
    import pandas as pd

    df = pd.read_csv("dataset/classification/train.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    res = run_nested_cv(
        estimator=svm.SVC(),
        X=X,
        y=y,
        param_grid={"kernel": ["linear", "rbf"], "C": [0.1, 1.0]},
        inner_cv=3,
        outer_cv=4,
        scoring="accuracy",
        task="classification",
    )
    assert len(res.outer_scores) == 4
    assert isinstance(res.mean_score, float)
    assert isinstance(res.best_params_per_fold, list)
