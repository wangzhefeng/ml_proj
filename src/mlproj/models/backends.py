from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA, LatentDirichletAllocation, NMF, PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import LocalOutlierFactor

from .adapters import UniversalModelAdapter


@dataclass
class TopicModelWrapper:
    method: str = "lda"
    vectorizer_type: str = "count"
    vectorizer_params: dict[str, Any] | None = None
    model_params: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        vec_params = dict(self.vectorizer_params or {})
        model_params = dict(self.model_params or {})

        if self.vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(**vec_params)
        else:
            self.vectorizer = CountVectorizer(**vec_params)

        if self.method == "nmf":
            self.model = NMF(**model_params)
        else:
            self.model = LatentDirichletAllocation(**model_params)

    def fit(self, X, y=None):
        texts = _to_text_iterable(X)
        mat = self.vectorizer.fit_transform(texts)
        self.model.fit(mat)
        return self

    def transform(self, X):
        texts = _to_text_iterable(X)
        mat = self.vectorizer.transform(texts)
        return self.model.transform(mat)

    def predict(self, X):
        topic_dist = self.transform(X)
        return np.argmax(topic_dist, axis=1)

    def get_params(self, deep: bool = False):
        return {
            "method": self.method,
            "vectorizer_type": self.vectorizer_type,
            "vectorizer_params": self.vectorizer_params or {},
            "model_params": self.model_params or {},
        }


def _to_text_iterable(X) -> list[str]:
    if hasattr(X, "columns"):
        if "text" in X.columns:
            return X["text"].astype(str).tolist()
        first_col = X.columns[0]
        return X[first_col].astype(str).tolist()
    arr = np.asarray(X)
    if arr.ndim == 1:
        return [str(v) for v in arr]
    return [str(v[0]) for v in arr]


def build_sklearn_adapter(task: str, model_name: str, params: dict[str, Any]) -> UniversalModelAdapter:
    if task == "classification":
        if model_name == "logistic_regression":
            model = LogisticRegression(max_iter=1000, **params)
        elif model_name == "random_forest":
            model = RandomForestClassifier(random_state=42, **params)
        else:
            raise ValueError(
                f"Unknown sklearn model '{model_name}' for task '{task}'"
            )

    elif task == "regression":
        if model_name == "linear_regression":
            model = LinearRegression(**params)
        elif model_name == "random_forest":
            model = RandomForestRegressor(random_state=42, **params)
        else:
            raise ValueError(
                f"Unknown sklearn model '{model_name}' for task '{task}'"
            )

    elif task == "clustering":
        if model_name == "kmeans":
            model = KMeans(random_state=42, **params)
        elif model_name == "kmeans_small":
            model = KMeans(n_clusters=3, random_state=42, **params)
        else:
            raise ValueError(
                f"Unknown sklearn model '{model_name}' for task '{task}'"
            )

    elif task == "pca_reduction":
        if model_name == "pca":
            model = PCA(random_state=42, **params)
        elif model_name == "incremental_pca":
            model = IncrementalPCA(**params)
        else:
            raise ValueError(
                f"Unknown sklearn model '{model_name}' for task '{task}'"
            )

    elif task == "anomaly_detection":
        if model_name == "isolation_forest":
            model = IsolationForest(random_state=42, **params)
        elif model_name == "local_outlier_factor":
            lof_params = {"novelty": True, **params}
            model = LocalOutlierFactor(**lof_params)
        else:
            raise ValueError(
                f"Unknown sklearn model '{model_name}' for task '{task}'"
            )

    elif task == "topic_modeling":
        if model_name == "lda":
            model = TopicModelWrapper(method="lda", model_params=params)
        elif model_name == "nmf":
            model = TopicModelWrapper(method="nmf", model_params=params)
        else:
            raise ValueError(
                f"Unknown sklearn model '{model_name}' for task '{task}'"
            )
    else:
        raise ValueError(f"Unsupported task: {task}")

    capabilities = {
        "predict" if hasattr(model, "predict") else "",
        "predict_proba" if hasattr(model, "predict_proba") else "",
        "decision_function" if hasattr(model, "decision_function") else "",
        "transform" if hasattr(model, "transform") else "",
        "score_samples" if hasattr(model, "score_samples") else "",
    }
    capabilities.discard("")
    return UniversalModelAdapter(
        model=model,
        backend="sklearn",
        model_name=model_name,
        capabilities=capabilities,
    )


def build_lightgbm_adapter(task: str, model_name: str, params: dict[str, Any]) -> UniversalModelAdapter:
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("lightgbm backend is not available") from exc

    if task == "classification":
        model = lgb.LGBMClassifier(**params)
    elif task == "regression":
        model = lgb.LGBMRegressor(**params)
    else:
        raise ValueError(f"lightgbm backend does not support task '{task}'")

    capabilities = {"predict", "predict_proba"}
    return UniversalModelAdapter(model=model, backend="lightgbm", model_name=model_name, capabilities=capabilities)


def build_xgboost_adapter(task: str, model_name: str, params: dict[str, Any]) -> UniversalModelAdapter:
    try:
        import xgboost as xgb
    except Exception as exc:
        raise RuntimeError("xgboost backend is not available") from exc

    if task == "classification":
        model = xgb.XGBClassifier(**params)
    elif task == "regression":
        model = xgb.XGBRegressor(**params)
    else:
        raise ValueError(f"xgboost backend does not support task '{task}'")

    capabilities = {"predict", "predict_proba"}
    return UniversalModelAdapter(model=model, backend="xgboost", model_name=model_name, capabilities=capabilities)


def build_catboost_adapter(task: str, model_name: str, params: dict[str, Any]) -> UniversalModelAdapter:
    try:
        import catboost as cab
    except Exception as exc:
        raise RuntimeError("catboost backend is not available") from exc

    if task == "classification":
        model = cab.CatBoostClassifier(verbose=False, **params)
    elif task == "regression":
        model = cab.CatBoostRegressor(verbose=False, **params)
    else:
        raise ValueError(f"catboost backend does not support task '{task}'")

    capabilities = {"predict", "predict_proba"}
    return UniversalModelAdapter(model=model, backend="catboost", model_name=model_name, capabilities=capabilities)
