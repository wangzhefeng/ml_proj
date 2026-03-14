# -*- coding: utf-8 -*-

# ***************************************************
# * File        : pipeline.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-18
# * Version     : 1.0.091823
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import pandas as pd

# utils
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
# 缺失值填充
from sklearn.impute import SimpleImputer, MissingIndicator
# 数据预处理
from sklearn.preprocessing import (
    StandardScaler,
    scale,
    MinMaxScaler,
    minmax_scale,
    MaxAbsScaler,
    maxabs_scale,
    RobustScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
    KBinsDiscretizer,
    Binarizer,
    PolynomialFeatures,
    FunctionTransformer,
    OrdinalEncoder, 
    OneHotEncoder,
)
# model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import linearSVC, SVC
from sklearn.decomposition import PCA
# model selection
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV, 
    RandomizedSearchCV, 
    ParameterGrid
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data
# ------------------------------
data_url = ("https://raw.githubusercontent.com/amueller/scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv")
data = pd.read_csv(data_url)
X_train, X_test, y_train, y_test = train_test_split(data, train_size = 0.8, random_state = 2023)

# ------------------------------
# 数值特征转换与处理
# ------------------------------
# number features
numeric_features = []
numeric_transformer = [
    # 缺失值填充
    ("missing_imputer_mean", SimpleImputer(missing_values = np.nan, strategy = "mean")),
    ("missing_imputer_median", SimpleImputer(missing_values = np.nan, strategy = "median")),
    ("missing_imputer_mode", SimpleImputer(missing_values = np.nan, strategy = "most_frequent")),
    ("missing_imputer_constant", SimpleImputer(missing_values = -1, strategy = "constant", fill_value = 0)),
    # 转换
    ("stanadard_scaler", StandardScaler()),
    ("min_max_scaler", MinMaxScaler(feature_range = (0, 1))),
    ("max_abs_scaler", MaxAbsScaler()),
    ("robust_scaler", RobustScaler(quantile_range = (25, 75))),
    ("power_scaler", PowerTransformer(method = "yeo-johnson")),
    ("power_scaler", PowerTransformer(method = "box-cox")),
    ("quantile_scaler", QuantileTransformer(output_distribution = "normal")),
    ("uniform_quantile_scaler", QuantileTransformer(output_distributions = "uniform")),
    ("normalizer_scaler", Normalizer()),
    # 二值化
    ("binning", KBinsDiscretizer(n_bins = [], encode = "ordinal")),
    ("binarizer", Binarizer())
]
# ------------------------------
# 类别特征转换与处理
# ------------------------------
cate_features = []
cate_transformer = [
    ("missing_imputer_mean", SimpleImputer(missing_values = np.nan, strategy = "mean")),
    ("missing_imputer_median", SimpleImputer(missing_values = np.nan, strategy = "median")),
    ("missing_imputer_mode", SimpleImputer(missing_values = np.nan, strategy = "most_frequent")),
    ("missing_imputer_constant", SimpleImputer(missing_values = -1, strategy = "constant", fill_value = 0)),
]
# ------------------------------
# 降维
# ------------------------------

# ------------------------------
# 特征工程
# ------------------------------

# ------------------------------
# 模型超参数调整
# ------------------------------
clf = None

# method 1
param_grid = {
    "perprocessor_num_imputer_strategy": ["mean", "median"],
    "classifier_c": [0.1, 1.0, 10, 100],
}
grid_search_1 = GridSearchCV(clf, param_grid, cv = 10, iid = False)
grid_search_1.fit(X_train, y_train)


# method 2
param_grid = dict(
    reduce_dim__n_components = [2, 5, 10],
    clf__C = [0.1, 10, 100]
)
pipeline = Pipeline(
    clf, 
    numeric_transformer,
    cate_transformer,
)
grid_search_2 = GridSearchCV(pipeline, param_grid = param_grid)
grid_search_1.fit(X_train, y_train)

# ------------------------------
# Pipeline
# ------------------------------
estimators = [
    ("PCA_reduce_dim", PCA()),
    ("SVC_clf", SVC())
]
pipeline = Pipeline(steps = estimators)
pipeline_man = make_pipeline(
    Binarizer(),
)
print(pipeline.setps[0])
print(pipeline.named_steps["PCA_reduce_dim"])
print(pipeline.set_params(SVC_clf__C = 10))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
