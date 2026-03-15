import pandas as pd
import pytest

from mlproj.models.factory import create_model


def test_classification_and_regression_model_creation():
    df = pd.read_csv("dataset/classification/train.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    clf = create_model("classification", "logistic_regression")
    clf.fit(X, y)
    pred = clf.predict(X.head(3))
    assert len(pred) == 3

    reg_df = pd.read_csv("dataset/regression/train.csv")
    Xr = reg_df.drop(columns=["target"])
    yr = reg_df["target"]
    reg_model = create_model("regression", "linear_regression")
    reg_model.fit(Xr, yr)
    reg_pred = reg_model.predict(Xr.head(2))
    assert len(reg_pred) == 2


def test_invalid_model_name_raises():
    with pytest.raises(ValueError):
        create_model("classification", "unknown_model")
