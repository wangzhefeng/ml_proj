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
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    y = digits.target

    tuned_parameters = [
        {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
    ]

    res = run_grid_search(
        svm.SVC(), X, y, param_grid=tuned_parameters, cv=5, scoring="precision_macro"
    )
    print("best_params:", res.best_params)
    print("best_score:", res.best_score)


if __name__ == "__main__":
    main()
