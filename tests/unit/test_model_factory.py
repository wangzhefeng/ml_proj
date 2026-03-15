from sklearn.datasets import load_iris

from mlproj.models.factory import create_model


def test_model_factory_classification_fit_predict():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target

    model = create_model("classification", "logistic_regression")
    model.fit(X, y)
    pred = model.predict(X.head(5))

    assert len(pred) == 5
