from __future__ import annotations

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

try:
    from mlproj.selection.search import run_halving_grid_search
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.selection.search import run_halving_grid_search


def main() -> None:
    X, y = datasets.make_classification(
        n_samples=240,
        n_features=10,
        n_informative=6,
        random_state=0,
    )
    estimator = RandomForestClassifier(random_state=0)
    param_grid = {
        "max_depth": [3, None],
        "min_samples_split": [2, 4],
        "criterion": ["gini", "entropy"],
    }

    res = run_halving_grid_search(
        estimator=estimator,
        X=X,
        y=y,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        factor=2,
    )

    print("best_estimator:", res.best_estimator)
    print("best_score:", res.best_score)
    print("best_params:", res.best_params)


if __name__ == "__main__":
    main()
