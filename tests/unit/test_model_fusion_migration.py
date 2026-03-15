import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mlproj.fusion import (
    average_predictions,
    build_adaboost_classifier,
    build_bagging_classifier,
    build_stacking_classifier,
    build_voting_classifier,
    rank_models_by_score,
)
from mlproj.legacy_models import run_fusion_legacy_demo


def test_fusion_builders_trainable():
    df = pd.read_csv("dataset/classification/train.csv")
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = [
        build_voting_classifier(),
        build_stacking_classifier(),
        build_bagging_classifier(n_estimators=4),
        build_adaboost_classifier(),
    ]

    for model in models:
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        assert 0.0 <= float(score) <= 1.0


def test_average_and_sort_helpers():
    avg = average_predictions(
        [
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
        ]
    )
    assert np.allclose(avg, np.array([0.5, 1.5, 2.5]))

    ranked = rank_models_by_score({"a": 0.6, "b": 0.8, "c": 0.7})
    assert ranked[0][0] == "b"


def test_fusion_legacy_bridge_scripts():
    for script_name in [
        "model_fusion/voting.py",
        "model_fusion/stacking.py",
        "model_fusion/baggging.py",
        "model_fusion/boosting.py",
        "model_fusion/averaging.py",
        "model_fusion/sorting.py",
        "model_fusion/PyStackNet.py",
    ]:
        out = run_fusion_legacy_demo(script_name)
        assert "method" in out
