from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectPercentile, VarianceThreshold, chi2
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import (
    Binarizer,
    KBinsDiscretizer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
)
from sklearn.svm import LinearSVC


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


@dataclass
class FeatureBuilding:
    def gen_polynomial_features(
        self,
        data: pd.DataFrame | np.ndarray,
        degree: int = 2,
        is_interaction_only: bool = True,
        is_include_bias: bool = True,
    ) -> np.ndarray:
        pf = PolynomialFeatures(
            degree=degree,
            interaction_only=is_interaction_only,
            include_bias=is_include_bias,
        )
        return pf.fit_transform(data)


class CategoryFeatureEncoder:
    @staticmethod
    def value_counts_encode(series: pd.Series) -> pd.Series:
        counts = series.value_counts(dropna=False)
        return series.map(counts).astype(float)


def binarization(feature: pd.Series, threshold: float = 0.0, is_copy: bool = True) -> np.ndarray:
    transfer = Binarizer(threshold=threshold, copy=is_copy)
    transformed_data = transfer.fit_transform(np.asarray(feature).reshape(-1, 1))
    return transformed_data.reshape(-1)


def kbins(
    feature: pd.Series,
    n_bins: int,
    encode: str = "ordinal",
    strategy: str = "quantile",
) -> np.ndarray:
    transfer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    return transfer.fit_transform(np.asarray(feature).reshape(-1, 1))


def one_hot_encode_low_cardinality(data: pd.DataFrame, limit_value: int = 10) -> pd.DataFrame:
    feature_cnt = data.shape[1]
    class_df = pd.DataFrame(index=data.index)
    normal_index: list[int] = []

    for i in range(feature_cnt):
        if data.iloc[:, i].nunique(dropna=False) < limit_value:
            dummies = pd.get_dummies(data.iloc[:, i], prefix=data.columns[i], dummy_na=True)
            class_df = pd.concat([class_df, dummies], axis=1)
        else:
            normal_index.append(i)
    return pd.concat([data.iloc[:, normal_index], class_df], axis=1)


def one_hot_encoder(feature: pd.DataFrame | np.ndarray):
    enc = OneHotEncoder(categories="auto", handle_unknown="ignore")
    return enc.fit_transform(feature)


def order_encoder(feature: pd.DataFrame | np.ndarray):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return enc.fit_transform(feature)


def label_encoder(data: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    out = data.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            out[c] = le.fit_transform(out[c].astype(str))
    return out


def split_text_feature_column(series: pd.Series, sep: str = "_", index: int = 0) -> pd.Series:
    return series.astype(str).str.split(sep).str[index]


def skewed_features(data: pd.DataFrame, num_feat_idx, limit_value: float = 0.75) -> pd.Index:
    skewed_feat_values = data[num_feat_idx].apply(lambda x: skew(x.dropna()))
    skewed_feat_values = skewed_feat_values[np.abs(skewed_feat_values) > limit_value]
    return skewed_feat_values.index


def numeric_categorical_split(
    data: pd.DataFrame,
    limit_value: int = 0,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, list[str]]:
    num_feat_idx: list[str] = []
    cate_feat_idx: list[str] = []
    for col in data.columns:
        if (
            pd.api.types.is_numeric_dtype(data[col])
            and data[col].nunique(dropna=False) >= limit_value
        ):
            num_feat_idx.append(col)
        else:
            cate_feat_idx.append(col)
    return data[num_feat_idx], num_feat_idx, data[cate_feat_idx], cate_feat_idx


def nan_feature_remove(data: pd.DataFrame, rate_base: float = 0.4):
    all_cnt = data.shape[0]
    available_index: list[int] = []
    for i in range(data.shape[1]):
        rate = np.isnan(np.asarray(data.iloc[:, i])).sum() / all_cnt
        if rate <= rate_base:
            available_index.append(i)
    return data.iloc[:, available_index], available_index


def low_variance_feature_remove(data: pd.DataFrame | np.ndarray, rate_base: float = 0.0):
    sel = VarianceThreshold(threshold=rate_base)
    return sel.fit_transform(data)


def col_filter(
    mtx_train: np.ndarray,
    y_train: np.ndarray,
    mtx_test: np.ndarray,
    func=chi2,
    percentile: int = 90,
):
    feature_select = SelectPercentile(func, percentile=percentile)
    feature_select.fit(mtx_train, y_train)
    return feature_select.transform(mtx_train), feature_select.transform(mtx_test)


def model_based_feature_selection(
    data: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    model: str = "tree",
    n_estimators: int = 50,
):
    if model == "tree":
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42).fit(data, target)
    elif model == "svm":
        clf = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data, target)
    elif model == "lr":
        clf = LogisticRegression(C=0.01, penalty="l1", solver="liblinear").fit(data, target)
    elif model == "lasso":
        clf = Lasso(alpha=0.001, random_state=42).fit(data, target)
    else:
        raise ValueError("model must be one of {'tree', 'svm', 'lr', 'lasso'}")

    selector = SelectFromModel(clf, prefit=True)
    return selector.transform(data)


# Backward-compatible aliases
NumericCategoricalSplit = numeric_categorical_split
SkewedFeatures = skewed_features
oneHotEncoding = one_hot_encode_low_cardinality
