from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_iris, load_wine
from sklearn.svm import SVC

from mlproj.config import load_config
from mlproj.data.loader import DatasetLoader
from mlproj.features.pipeline import FeaturePipeline
from mlproj.preprocess.base import SklearnPreprocessor
from mlproj.selection.search import run_random_search
from mlproj.training.trainer import Trainer


def run_lgb_clf_legacy_demo(
    config_path: str = "configs/classification/train.yaml",
) -> dict[str, str]:
    cfg = load_config(config_path)
    artifact = Trainer(artifact_root=cfg.get("artifact_root", "artifacts")).train(cfg)
    return {
        "run_id": artifact.run_id,
        "model_uri": str(artifact.model_uri),
        "metrics_uri": str(artifact.metrics_uri),
    }


def run_featuretools_legacy_demo() -> dict[str, int]:
    data = load_wine(as_frame=True).data.head(30)
    pre = SklearnPreprocessor().fit(data)
    transformed = pre.transform(data)
    feats = FeaturePipeline().fit(transformed).transform(transformed)
    return {
        "rows": int(feats.shape[0]),
        "cols": int(feats.shape[1]),
    }


def run_sklearn_legacy_demo(config_path: str = "configs/regression/train.yaml") -> dict[str, str]:
    cfg = load_config(config_path)
    artifact = Trainer(artifact_root=cfg.get("artifact_root", "artifacts")).train(cfg)
    return {
        "run_id": artifact.run_id,
        "metrics_uri": str(artifact.metrics_uri),
    }


def run_optuna_legacy_demo() -> dict[str, object]:
    iris = load_iris()
    distributions = {
        "kernel": ["linear", "rbf"],
        "C": np.linspace(0.5, 5.0, num=10),
    }
    result = run_random_search(
        estimator=SVC(),
        X=iris.data,
        y=iris.target,
        param_distributions=distributions,
        n_iter=5,
        cv=3,
        scoring="accuracy",
        random_state=42,
    )
    return {
        "best_params": result.best_params,
        "best_score": result.best_score,
    }


def run_pipeline_legacy_demo(
    config_path: str = "configs/classification/train.yaml",
) -> dict[str, int]:
    cfg = load_config(config_path)
    ds = DatasetLoader(random_state=42).load(cfg)
    pre = SklearnPreprocessor().fit(ds.X_train, ds.y_train)
    x_train = pre.transform(ds.X_train)
    feats = FeaturePipeline().fit(x_train, ds.y_train).transform(x_train)
    return {
        "train_rows": int(feats.shape[0]),
        "train_cols": int(feats.shape[1]),
    }


def run_pls_legacy_demo() -> np.ndarray:
    x = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [2.0, 5.0, 4.0],
        ]
    )
    y = np.array(
        [
            [0.1, -0.2],
            [0.9, 1.1],
            [6.2, 5.9],
            [11.9, 12.3],
        ]
    )
    model = PLSRegression(n_components=2)
    model.fit(x, y)
    return model.predict(x)


def run_quadratic_legacy_demo() -> dict[str, float]:
    xs = np.linspace(-10, 10, 401)
    ys = np.array([-1.0, 0.0, 1.0])
    best = None
    for x in xs:
        for y in ys:
            value = x**2 + y
            if best is None or value < best["best_value"]:
                best = {
                    "best_value": float(value),
                    "best_x": float(x),
                    "best_y": float(y),
                }
    return best or {"best_value": 0.0, "best_x": 0.0, "best_y": 0.0}
