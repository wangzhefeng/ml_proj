import pandas as pd

from mlproj.analysis.causal import estimate_ate_linear
from mlproj.analysis.factor_analysis import run_factor_analysis
from mlproj.analysis.hypothesis import corr_test
from mlproj.analysis.multiple_regression import run_multiple_regression


def test_multiple_regression_new_interface():
    df = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [2, 3, 4, 5], "y": [2, 4, 6, 8]})
    res = run_multiple_regression(df, ["x1", "x2"], "y", regul="linear")
    assert set(res.columns) == {"x1", "x2"}


def test_corr_test_new_interface():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})
    res = corr_test(df, ["a"], ["b"], methods=("pearson",))
    assert len(res) == 1
    assert "corr_coef" in res.columns


def test_factor_analysis_new_interface():
    df = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [2, 3, 4, 5], "f3": [3, 4, 5, 6]})
    out = run_factor_analysis(df, n_factors=2)
    assert "loadings" in out
    assert out["loadings"].shape[1] == 2


def test_causal_new_interface():
    df = pd.DataFrame(
        {
            "treatment": [0, 1, 0, 1, 0, 1],
            "w1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "outcome": [1.0, 3.1, 1.5, 3.8, 1.9, 4.2],
        }
    )
    ate = estimate_ate_linear(df, "treatment", "outcome", ["w1"])
    assert isinstance(ate, float)
