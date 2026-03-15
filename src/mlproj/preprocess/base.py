from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "Preprocessor": ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class SklearnPreprocessor:
    transformer: ColumnTransformer | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SklearnPreprocessor":
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        self.transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, categorical_cols),
            ],
            remainder="drop",
        )
        self.transformer.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.transformer is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        transformed = self.transformer.transform(X)
        columns = self.get_feature_names()
        return pd.DataFrame(transformed, columns=columns, index=X.index)

    def get_feature_names(self) -> list[str]:
        if self.transformer is None:
            return []
        names: list[str] = []
        for name, transformer, cols in self.transformer.transformers_:
            if name == "remainder":
                continue
            if not cols:
                continue
            if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
                ohe = transformer.named_steps["onehot"]
                feature_names = ohe.get_feature_names_out(cols).tolist()
                names.extend(feature_names)
            else:
                names.extend(cols)
        return names
