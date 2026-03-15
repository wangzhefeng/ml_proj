import pandas as pd

from mlproj.models.factory import create_model


def test_model_factory_classification_fit_predict():
    df = pd.read_csv("dataset/classification/train.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    model = create_model("classification", "logistic_regression")
    model.fit(X, y)
    pred = model.predict(X.head(5))

    assert len(pred) == 5
