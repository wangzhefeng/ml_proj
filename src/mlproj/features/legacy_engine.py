from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectPercentile, VarianceThreshold, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    normalize,
)
from sklearn.svm import LinearSVC
from scipy.stats import skew


def binarization(feature: pd.Series, threshold: float = 0.0, is_copy: bool = True) -> np.ndarray:
    transfer = Binarizer(threshold=threshold, copy=is_copy)
    transformed_data = transfer.fit_transform(np.array(feature).reshape(-1, 1))
    return transformed_data.reshape(1, -1)[0]


def kbins(feature: pd.Series, n_bins: int, encode: str = "ordinal", strategy: str = "quantile"):
    transfer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    return transfer.fit_transform(np.array(feature).reshape(-1, 1))


@dataclass
class FeatureBuilding:
    def gen_polynomial_features(
        self,
        data,
        degree: int = 2,
        is_interaction_only: bool = True,
        is_include_bias: bool = True,
    ):
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


def oneHotEncoding(data: pd.DataFrame, limit_value: int = 10) -> pd.DataFrame:
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


def one_hot_encoder(feature):
    enc = OneHotEncoder(categories="auto", handle_unknown="ignore")
    return enc.fit_transform(feature)


def order_encoder(feature):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return enc.fit_transform(feature)


def label_encoder(data: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    out = data.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            out[c] = le.fit_transform(out[c].astype(str))
    return out


class pca:
    def __init__(
        self,
        X,
        n_components,
        whiten: bool = False,
        copy: bool = True,
        svd_solver: str = "auto",
        tol: float = 0.0,
        iterated_power: str = "auto",
        random_state: int | None = None,
        batch_size: int | None = None,
    ):
        self.X = X
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.batch_size = batch_size

    def pca(self):
        from sklearn.decomposition import PCA

        pca_model = PCA(
            n_components=self.n_components,
            copy=self.copy,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            random_state=self.random_state,
        )
        pca_model.fit_transform(self.X)
        return pca_model

    def incremental_pca(self):
        from sklearn.decomposition import IncrementalPCA

        pca_model = IncrementalPCA(
            n_components=self.n_components,
            whiten=self.whiten,
            copy=self.copy,
            batch_size=self.batch_size,
        )
        pca_model.fit_transform(self.X)
        return pca_model


class lda:
    def __init__(self, X, y, n_components):
        self.X = X
        self.y = y
        self.n_components = n_components

    def lda(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        model = LinearDiscriminantAnalysis(n_components=self.n_components)
        model.fit_transform(self.X, self.y)
        return model


def nan_feature_remove(data: pd.DataFrame, rate_base: float = 0.4):
    all_cnt = data.shape[0]
    available_index: list[int] = []
    for i in range(data.shape[1]):
        rate = np.isnan(np.array(data.iloc[:, i])).sum() / all_cnt
        if rate <= rate_base:
            available_index.append(i)
    return data.iloc[:, available_index], available_index


def low_variance_feature_remove(data, rate_base: float = 0.0):
    sel = VarianceThreshold(threshold=rate_base)
    return sel.fit_transform(data)


def col_filter(mtx_train, y_train, mtx_test, func=chi2, percentile: int = 90):
    feature_select = SelectPercentile(func, percentile=percentile)
    feature_select.fit(mtx_train, y_train)
    return feature_select.transform(mtx_train), feature_select.transform(mtx_test)


def model_based_feature_selection(data, target, model: str = "tree", n_estimators: int = 50):
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


def NumericCategoricalSplit(data: pd.DataFrame, limit_value: int = 0):
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


class Stacking:
    def __init__(
        self, clf, train_x, train_y, test_x, clf_name: str, folds: int, label_split=None
    ) -> None:
        self.train_x = np.asarray(train_x)
        self.train_y = np.asarray(train_y)
        self.test_x = np.asarray(test_x)
        self.clf = clf
        self.clf_name = clf_name
        self.folds = int(folds)
        self.label_split = label_split
        self.train = np.zeros((self.train_x.shape[0], 1))
        self.test = np.zeros((self.test_x.shape[0], 1))
        self.test_pred = np.empty((self.folds, self.test_x.shape[0], 1))

    def sklearn_reg(self):
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=0)
        cv_scores = []
        for i, (train_index, test_index) in enumerate(kf.split(self.train_x, self.label_split)):
            tr_x, tr_y = self.train_x[train_index], self.train_y[train_index]
            te_x, te_y = self.train_x[test_index], self.train_y[test_index]

            self.clf.fit(tr_x, tr_y)
            pred = self.clf.predict(te_x).reshape(-1, 1)
            self.train[test_index] = pred
            self.test_pred[i, :] = self.clf.predict(self.test_x).reshape(-1, 1)
            cv_scores.append(float(mean_squared_error(te_y, pred.ravel())))
        self.test[:] = self.test_pred.mean(axis=0)
        return self.train.reshape(-1, 1), self.test.reshape(-1, 1)

    def sklearn_clf(self):
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=0)
        cv_scores = []
        for i, (train_index, test_index) in enumerate(kf.split(self.train_x, self.label_split)):
            tr_x, tr_y = self.train_x[train_index], self.train_y[train_index]
            te_x, te_y = self.train_x[test_index], self.train_y[test_index]

            self.clf.fit(tr_x, tr_y)
            proba = self.clf.predict_proba(te_x)
            self.train[test_index] = proba[:, [0]]
            self.test_pred[i, :] = self.clf.predict_proba(self.test_x)[:, [0]]
            cv_scores.append(float(log_loss(te_y, proba)))
        self.test[:] = self.test_pred.mean(axis=0)
        return self.train.reshape(-1, 1), self.test.reshape(-1, 1)

    def xgb_reg(self):
        raise RuntimeError(
            "xgb_reg is not implemented in v1 bridge; use sklearn_reg or mlproj training pipeline"
        )

    def lgb_reg(self):
        raise RuntimeError(
            "lgb_reg is not implemented in v1 bridge; use sklearn_reg or mlproj training pipeline"
        )

    def xgb_clf(self):
        raise RuntimeError(
            "xgb_clf is not implemented in v1 bridge; use sklearn_clf or mlproj training pipeline"
        )

    def lgb_clf(self):
        raise RuntimeError(
            "lgb_clf is not implemented in v1 bridge; use sklearn_clf or mlproj training pipeline"
        )


def split_text_feature_column(series: pd.Series, sep: str = "_", index: int = 0) -> pd.Series:
    return series.astype(str).str.split(sep).str[index]


def SkewedFeatures(data: pd.DataFrame, num_feat_idx, limit_value: float = 0.75) -> pd.Index:
    skewed_feat_values = data[num_feat_idx].apply(lambda x: skew(x.dropna()))
    skewed_feat_values = skewed_feat_values[np.abs(skewed_feat_values) > limit_value]
    return skewed_feat_values.index


def random_forest_importance_classifier(x_train, y_train, feature_labels, threshold: float = 0.15):
    rf_clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    rf_clf.fit(x_train, y_train)
    importances = rf_clf.feature_importances_
    selected_mask = importances > threshold
    x_selected = x_train[:, selected_mask]
    selected_features = [feature_labels[i] for i, keep in enumerate(selected_mask) if keep]
    return {
        "x_selected": x_selected,
        "importances": importances,
        "selected_features": selected_features,
    }


def random_forest_importance_regressor(x_train, y_train, feature_labels, threshold: float = 0.15):
    rf_reg = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
    rf_reg.fit(x_train, y_train)
    importances = rf_reg.feature_importances_
    selected_mask = importances > threshold
    x_selected = x_train[:, selected_mask]
    selected_features = [feature_labels[i] for i, keep in enumerate(selected_mask) if keep]
    return {
        "x_selected": x_selected,
        "importances": importances,
        "selected_features": selected_features,
    }


class MissingPreprocessing:
    def __init__(self, feature):
        self.feature = feature

    def simple_imputer(self):
        return SimpleImputer().fit_transform(self.feature)

    def QuantileImpute(self, data_input: pd.DataFrame, key_value: float = 0.95):
        data_union = pd.DataFrame(index=data_input.index)
        for col in data_input.columns:
            key = data_input[col].dropna().quantile(key_value)
            col_data = data_input[col].fillna(value=key).clip(upper=key)
            data_union[col] = col_data
        return data_union

    def ValueImpute(self, data_input: pd.DataFrame, Value: float):
        data_union = pd.DataFrame(index=data_input.index)
        for col in data_input.columns:
            col_data = data_input[col].fillna(value=Value).clip(upper=Value)
            data_union[col] = col_data
        return data_union

    def ModeImpute(self, data_input: pd.DataFrame, key_value: float = 0.95):
        data_union = pd.DataFrame(index=data_input.index)
        for col in data_input.columns:
            col_non_na = data_input[col].dropna()
            mode_value = col_non_na.mode().iloc[0] if not col_non_na.empty else 0
            upper = col_non_na.quantile(key_value) if not col_non_na.empty else mode_value
            col_data = data_input[col].clip(upper=upper).fillna(value=mode_value)
            data_union[col] = col_data
        return data_union

    def nan_fill(
        self, data: pd.DataFrame, limit_value: int = 10, continuous_dealed_method: str = "mean"
    ):
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
            [data.iloc[:, normal_index], continuous_feature_df, class_feature_df], axis=1
        )


def NormalityTransform(feature):
    return np.log1p(np.asarray(feature))


def standard_center(features, is_copy: bool = True, with_mean: bool = True, with_std: bool = True):
    ss = StandardScaler(copy=is_copy, with_mean=with_mean, with_std=with_std)
    return ss.fit_transform(features)


def normalizer_min_max(features):
    return MinMaxScaler().fit_transform(features)


def normalizer_min_max_feature(feature):
    feat = np.asarray(feature)
    return (feat - feat.min()) / (feat.max() - feat.min() + 1e-12)


def normalizer_L2(features):
    return Normalizer().fit_transform(features)


def normalizer_Ln(
    features, norm="l2", axis: int = 1, is_copy: bool = True, return_norm: bool = False
):
    return normalize(X=features, norm=norm, axis=axis, copy=is_copy, return_norm=return_norm)


def robust_tansform(features):
    return RobustScaler().fit_transform(features)


def log_transform_feature(feature):
    return np.log1p(feature)


def log1p_transform(features):
    ft = FunctionTransformer(np.log1p, validate=False)
    return ft.fit_transform(features)


def box_cox_transform(features):
    bc = PowerTransformer(method="box-cox", standardize=False)
    return bc.fit_transform(features)


def yeo_johnson_transform(features):
    yj = PowerTransformer(method="yeo-johnson", standardize=False)
    return yj.fit_transform(features)


def ploynomial_transform(features):
    return PolynomialFeatures().fit_transform(features)


def quantileNorm(feature):
    return QuantileTransformer(output_distribution="normal", random_state=0).fit_transform(feature)


def quantileUniform(feature, feat_test=None):
    qu = QuantileTransformer(random_state=0)
    feat_trans = qu.fit_transform(feature)
    if feat_test is None:
        return feat_trans
    return feat_trans, qu.transform(feat_test)


def downsample_majority(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    from sklearn.utils import resample

    df = X.copy()
    df["__target__"] = y.values
    counts = df["__target__"].value_counts()
    min_count = counts.min()

    parts = []
    for cls, _ in counts.items():
        cls_df = df[df["__target__"] == cls]
        parts.append(
            resample(cls_df, replace=False, n_samples=min_count, random_state=random_state)
        )

    sampled = pd.concat(parts).sample(frac=1.0, random_state=random_state)
    y_out = sampled.pop("__target__")
    return sampled, y_out


def simple_over_under_sample(
    X: pd.DataFrame, y: pd.Series, strategy: str = "downsample", random_state: int = 42
):
    strategy = strategy.lower()
    if strategy == "downsample":
        return downsample_majority(X, y, random_state=random_state)
    if strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE
        except Exception as exc:
            raise RuntimeError("imblearn is required for SMOTE strategy") from exc

        sampler = SMOTE(random_state=random_state)
        x_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(x_res, columns=X.columns), pd.Series(y_res)
    raise ValueError("strategy must be one of {'downsample', 'smote'}")
