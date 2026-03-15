from __future__ import annotations

from scipy.stats import randint
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

try:
    from mlproj.selection.search import run_halving_random_search
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.selection.search import run_halving_random_search


def main() -> None:
    X, y = datasets.make_classification(
        n_samples=300,
        n_features=12,
        n_informative=6,
        random_state=0,
    )

    estimator = RandomForestClassifier(n_estimators=40, random_state=0)
    param_dist = {
        "max_depth": [3, None],
        "max_features": randint(1, 6),
        "min_samples_split": randint(2, 8),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"],
    }

    res = run_halving_random_search(
        estimator=estimator,
        X=X,
        y=y,
        param_distributions=param_dist,
        cv=3,
        scoring="accuracy",
        factor=2,
        n_candidates=8,
        random_state=2024,
    )

    print("best_estimator:", res.best_estimator)
    print("best_score:", res.best_score)
    print("best_params:", res.best_params)


if __name__ == "__main__":
    main()
