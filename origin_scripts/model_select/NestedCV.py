from __future__ import annotations

from sklearn import datasets, svm

try:
    from mlproj.selection.search import run_nested_cv
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.selection.search import run_nested_cv


def main() -> None:
    iris = datasets.load_iris()
    estimator = svm.SVC()
    param_grid = {
        "kernel": ["linear", "rbf"],
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", "auto"],
    }

    res = run_nested_cv(
        estimator=estimator,
        X=iris.data,
        y=iris.target,
        param_grid=param_grid,
        inner_cv=3,
        outer_cv=4,
        scoring="accuracy",
        task="classification",
    )

    print("outer_scores:", res.outer_scores)
    print("mean_score:", res.mean_score)
    print("std_score:", res.std_score)
    print("best_params_per_fold:", res.best_params_per_fold)


if __name__ == "__main__":
    main()
