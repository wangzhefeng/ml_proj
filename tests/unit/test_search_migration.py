from sklearn import svm

from mlproj.selection.search import run_grid_search


def test_grid_search_bridge():
    import pandas as pd

    df = pd.read_csv("dataset/classification/train.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    res = run_grid_search(
        estimator=svm.SVC(),
        X=X,
        y=y,
        param_grid={"kernel": ["linear"], "C": [0.1, 1.0]},
        cv=3,
        scoring="accuracy",
    )
    assert isinstance(res.best_params, dict)
