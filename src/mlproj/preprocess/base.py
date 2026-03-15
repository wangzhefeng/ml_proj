from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, OneHotEncoder, RobustScaler, StandardScaler, normalize


class Preprocessor(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "Preprocessor": ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class IdentityPreprocessor:
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "IdentityPreprocessor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()
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


class MissingPreprocessing:
    def __init__(self, feature: pd.DataFrame):
        self.feature = feature

    def simple_imputer(self) -> np.ndarray:
        return SimpleImputer().fit_transform(self.feature)

    def quantile_impute(self, data_input: pd.DataFrame, key_value: float = 0.95) -> pd.DataFrame:
        data_union = pd.DataFrame(index=data_input.index)
        for col in data_input.columns:
            key = data_input[col].dropna().quantile(key_value)
            col_data = data_input[col].fillna(value=key).clip(upper=key)
            data_union[col] = col_data
        return data_union

    def value_impute(self, data_input: pd.DataFrame, value: float) -> pd.DataFrame:
        data_union = pd.DataFrame(index=data_input.index)
        for col in data_input.columns:
            col_data = data_input[col].fillna(value=value).clip(upper=value)
            data_union[col] = col_data
        return data_union

    def mode_impute(self, data_input: pd.DataFrame, key_value: float = 0.95) -> pd.DataFrame:
        data_union = pd.DataFrame(index=data_input.index)
        for col in data_input.columns:
            col_non_na = data_input[col].dropna()
            mode_value = col_non_na.mode().iloc[0] if not col_non_na.empty else 0
            upper = col_non_na.quantile(key_value) if not col_non_na.empty else mode_value
            col_data = data_input[col].clip(upper=upper).fillna(value=mode_value)
            data_union[col] = col_data
        return data_union

    def nan_fill(
        self,
        data: pd.DataFrame,
        limit_value: int = 10,
        continuous_dealed_method: str = "mean",
    ) -> pd.DataFrame:
        feature_cnt = data.shape[1]
        normal_index: list[int] = []
        continuous_feature_df = pd.DataFrame(index=data.index)
        class_feature_df = pd.DataFrame(index=data.index)

        for i in range(feature_cnt):
            col = data.iloc[:, i]
            if col.isna().any():
                nunique = col.nunique(dropna=True)
                if nunique >= limit_value:
                    if continuous_dealed_method == "mean":
                        continuous_feature_df[data.columns[i]] = col.fillna(col.mean())
                    elif continuous_dealed_method == "max":
                        continuous_feature_df[data.columns[i]] = col.fillna(col.max())
                    elif continuous_dealed_method == "min":
                        continuous_feature_df[data.columns[i]] = col.fillna(col.min())
                    else:
                        continuous_feature_df[data.columns[i]] = col.fillna(col.mean())
                elif 0 < nunique < limit_value:
                    dummies = pd.get_dummies(col.fillna("missing"), prefix=data.columns[i])
                    class_feature_df = pd.concat([class_feature_df, dummies], axis=1)
            else:
                normal_index.append(i)

        return pd.concat(
            [data.iloc[:, normal_index], continuous_feature_df, class_feature_df],
            axis=1,
        )

    # Legacy-style aliases
    QuantileImpute = quantile_impute
    ValueImpute = value_impute
    ModeImpute = mode_impute


def normality_transform(feature: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(feature))


def standard_center(
    features: pd.DataFrame | np.ndarray,
    is_copy: bool = True,
    with_mean: bool = True,
    with_std: bool = True,
) -> np.ndarray:
    ss = StandardScaler(copy=is_copy, with_mean=with_mean, with_std=with_std)
    return ss.fit_transform(features)


def normalizer_min_max(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    return MinMaxScaler().fit_transform(features)


def normalizer_min_max_feature(feature: pd.Series | np.ndarray) -> np.ndarray:
    feat = np.asarray(feature)
    return (feat - feat.min()) / (feat.max() - feat.min() + 1e-12)


def normalizer_l2(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    return Normalizer().fit_transform(features)


def normalizer_ln(
    features: pd.DataFrame | np.ndarray,
    norm: str = "l2",
    axis: int = 1,
    is_copy: bool = True,
    return_norm: bool = False,
):
    return normalize(X=features, norm=norm, axis=axis, copy=is_copy, return_norm=return_norm)


def robust_transform(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    return RobustScaler().fit_transform(features)


def log_transform_feature(feature: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(feature)


# Legacy-style aliases
NormalityTransform = normality_transform
normalizer_L2 = normalizer_l2
normalizer_Ln = normalizer_ln
robust_tansform = robust_transform



