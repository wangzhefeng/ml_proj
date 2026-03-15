import pandas as pd
import pytest

from mlproj.models.factory import create_model


def test_classification_and_timeseries_model_creation():
    df = pd.read_csv("dataset/classification/train.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    clf = create_model("classification", "logistic_regression")
    clf.fit(X, y)
    pred = clf.predict(X.head(3))
    assert len(pred) == 3

    ts_model = create_model("timeseries", "linear_regression")
    ts_model.fit(X, y)
    ts_pred = ts_model.predict(X.head(2))
    assert len(ts_pred) == 2


def test_invalid_model_name_raises():
    with pytest.raises(ValueError):
        create_model("classification", "unknown_model")
