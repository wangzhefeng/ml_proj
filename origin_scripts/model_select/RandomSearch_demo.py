from __future__ import annotations

from scipy.stats import uniform
from sklearn import datasets, svm

try:
    from mlproj.selection.search import run_random_search
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.selection.search import run_random_search


def main() -> None:
    iris = datasets.load_iris()
    distributions = {"kernel": ["linear", "rbf"], "C": uniform(loc=1, scale=9)}

    res = run_random_search(
        estimator=svm.SVC(),
        X=iris.data,
        y=iris.target,
        param_distributions=distributions,
        n_iter=4,
        cv=5,
        scoring="accuracy",
        random_state=2021,
    )

    print("best_estimator:", res.best_estimator)
    print("best_score:", res.best_score)
    print("best_params:", res.best_params)


if __name__ == "__main__":
    main()
