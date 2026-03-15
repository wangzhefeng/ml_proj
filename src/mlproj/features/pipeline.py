from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FeaturePipeline:
    selected_columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeaturePipeline":
        self.selected_columns = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_columns is None:
            raise RuntimeError("FeaturePipeline must be fitted before transform")
        missing = [c for c in self.selected_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns for inference: {missing}")
        return X[self.selected_columns].copy()
