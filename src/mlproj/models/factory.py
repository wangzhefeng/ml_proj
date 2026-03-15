from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


class ModelAdapter(Protocol):
    def fit(self, X: Any, y: Any | None = None) -> "ModelAdapter": ...

    def predict(self, X: Any) -> Any: ...

    def predict_proba(self, X: Any) -> Any: ...


@dataclass
class SklearnModelAdapter:
    model: Any

    def fit(self, X: Any, y: Any | None = None) -> "SklearnModelAdapter":
        if y is None:
            self.model.fit(X)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("Current model does not support predict_proba")


def create_model(
    task: str, model_name: str, params: dict[str, Any] | None = None
) -> SklearnModelAdapter:
    params = params or {}

    if task == "classification":
        if model_name == "logistic_regression":
            model = LogisticRegression(max_iter=1000, **params)
        elif model_name == "random_forest":
            model = RandomForestClassifier(random_state=42, **params)
        else:
            raise ValueError(
                f"Unknown model '{model_name}' for task '{task}'. Available: ['logistic_regression', 'random_forest']"
            )

    elif task == "regression":
        if model_name == "linear_regression":
            model = LinearRegression(**params)
        elif model_name == "random_forest":
            model = RandomForestRegressor(random_state=42, **params)
        else:
            raise ValueError(
                f"Unknown model '{model_name}' for task '{task}'. Available: ['linear_regression', 'random_forest']"
            )

    elif task == "clustering":
        if model_name == "kmeans":
            model = KMeans(random_state=42, **params)
        elif model_name == "kmeans_small":
            model = KMeans(n_clusters=3, random_state=42, **params)
        else:
            raise ValueError(
                f"Unknown model '{model_name}' for task '{task}'. Available: ['kmeans', 'kmeans_small']"
            )
    else:
        raise ValueError(f"Unsupported task: {task}")

    return SklearnModelAdapter(model=model)
