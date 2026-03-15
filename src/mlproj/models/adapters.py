from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class UniversalModelAdapter:
    model: Any
    backend: str
    model_name: str
    capabilities: set[str] = field(default_factory=set)

    def fit(self, X: Any, y: Any | None = None) -> "UniversalModelAdapter":
        if y is None:
            self.model.fit(X)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        if not hasattr(self.model, "predict"):
            raise AttributeError("Current model does not support predict")
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Current model does not support predict_proba")
        return self.model.predict_proba(X)

    def decision_function(self, X: Any) -> Any:
        if not hasattr(self.model, "decision_function"):
            raise AttributeError("Current model does not support decision_function")
        return self.model.decision_function(X)

    def transform(self, X: Any) -> Any:
        if not hasattr(self.model, "transform"):
            raise AttributeError("Current model does not support transform")
        return self.model.transform(X)

    def fit_predict(self, X: Any, y: Any | None = None) -> Any:
        if hasattr(self.model, "fit_predict"):
            if y is None:
                return self.model.fit_predict(X)
            return self.model.fit_predict(X, y)
        self.fit(X, y)
        return self.predict(X)

    def score_samples(self, X: Any) -> Any:
        if not hasattr(self.model, "score_samples"):
            raise AttributeError("Current model does not support score_samples")
        return self.model.score_samples(X)

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        if hasattr(self.model, "get_params"):
            return dict(self.model.get_params(deep=deep))
        return {}
