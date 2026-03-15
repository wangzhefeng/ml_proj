from __future__ import annotations

from sklearn import datasets, svm

try:
    from mlproj.selection.search import run_grid_search
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.selection.search import run_grid_search


def main() -> None:
    iris = datasets.load_iris()
    params = {"kernel": ["linear", "rbf"], "C": [1, 10]}

    res = run_grid_search(
        estimator=svm.SVC(),
        X=iris.data,
        y=iris.target,
        param_grid=params,
        cv=5,
        scoring="accuracy",
    )

    print("best_estimator:", res.best_estimator)
    print("best_score:", res.best_score)
    print("best_params:", res.best_params)


if __name__ == "__main__":
    main()
