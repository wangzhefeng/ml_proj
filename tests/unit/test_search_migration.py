from sklearn import datasets, svm

from mlproj.selection.search import run_grid_search


def test_grid_search_bridge():
    iris = datasets.load_iris()
    res = run_grid_search(
        estimator=svm.SVC(),
        X=iris.data,
        y=iris.target,
        param_grid={"kernel": ["linear"], "C": [0.1, 1.0]},
        cv=3,
        scoring="accuracy",
    )
    assert isinstance(res.best_params, dict)
