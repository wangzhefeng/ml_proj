from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelLifecycleHooks:
    def before_fit(self, model: Any, X: Any, y: Any | None = None) -> None:
        return None

    def after_fit(self, model: Any) -> None:
        return None

    def before_predict(self, model: Any, X: Any) -> None:
        return None

    def after_predict(self, model: Any, y_pred: Any) -> None:
        return None
