from __future__ import annotations

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

try:
    from mlproj.selection.search import run_grid_search, run_random_search
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.selection.search import run_grid_search, run_random_search


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target

    params = {
        "max_depth": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5],
    }

    estimator = RandomForestClassifier(random_state=0)
    grid_res = run_grid_search(
        estimator=estimator,
        X=X,
        y=y,
        param_grid=params,
        cv=3,
        scoring="accuracy",
    )
    print("[grid] best_score:", grid_res.best_score)
    print("[grid] best_params:", grid_res.best_params)

    random_res = run_random_search(
        estimator=estimator,
        X=X,
        y=y,
        param_distributions=params,
        n_iter=4,
        cv=3,
        scoring="accuracy",
        random_state=2023,
    )
    print("[random] best_score:", random_res.best_score)
    print("[random] best_params:", random_res.best_params)


if __name__ == "__main__":
    main()
