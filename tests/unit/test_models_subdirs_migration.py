import pandas as pd

from mlproj.models.factory import create_model


def test_supervised_classification_and_regression_dispatch():
    cls_df = pd.read_csv("dataset/classification/train.csv")
    X_cls = cls_df.drop(columns=["target"])
    y_cls = cls_df["target"]

    clf = create_model("classification", "random_forest", {"n_estimators": 10})
    clf.fit(X_cls, y_cls)
    cls_pred = clf.predict(X_cls.head(4))
    assert len(cls_pred) == 4

    reg_df = pd.read_csv("dataset/regression/train.csv")
    X_reg = reg_df.drop(columns=["target"])
    y_reg = reg_df["target"]

    reg = create_model("regression", "random_forest", {"n_estimators": 10})
    reg.fit(X_reg, y_reg)
    reg_pred = reg.predict(X_reg.head(4))
    assert len(reg_pred) == 4


def test_unsupervised_clustering_dispatch():
    clu_df = pd.read_csv("dataset/clustering/train.csv")
    label_col = "target" if "target" in clu_df.columns else "label"
    X = clu_df.drop(columns=[label_col])

    kmeans = create_model("clustering", "kmeans", {"n_clusters": 3})
    kmeans.fit(X)
    pred = kmeans.predict(X.head(5))
    assert len(pred) == 5
