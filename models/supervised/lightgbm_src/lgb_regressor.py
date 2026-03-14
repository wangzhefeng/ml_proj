# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lgb_regressor.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-19
# * Version     : 1.0.091900
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
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb
from dataset.reg_boston import get_boston

pd.set_option("display.max_columns", 50)
warnings.filterwarnings("ignore")
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data
# ------------------------------
# data split
boston = get_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target)
print(f"\nTrain/Test Sizes : {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")
# lgb data
train_dataset = lgb.Dataset(X_train, y_train, feature_name = boston.feature_names.tolist())
test_dataset = lgb.Dataset(X_test, y_test, feature_name = boston.feature_names.tolist())

# ------------------------------
# custom objective / loss function
# ------------------------------
def first_grad(predt, dmat):
    """
    Compute the first derivative for mean squared error
    """
    y = dmat.get_label() if isinstance(dmat, lgb.Dataset) else dmat
    return 2 * (y - predt)


def second_grad(predt, dmat):
    """
    Compute the second derivative for mean squared error
    """
    y = dmat.get_label() if isinstance(dmat, lgb.Dataset) else dmat
    return [1] * len(predt)


def mean_squared_error(predt, dmat):
    """
    Mean Squared Error function
    """
    predt[predt < -1] = -1 + 1e-6
    grad = first_grad(predt, dmat)
    hess = second_grad(predt, dmat)
    
    return grad, hess

# ------------------------------
# custom evaluation function
# ------------------------------
def mean_absolute_error(preds, dmat):
    actuals = dmat.get_label() if isinstance(dmat, lgb.Dataset) else dmat
    err = (actuals - preds).sum()
    is_higher_better = False

    return "MAE", err, is_higher_better

# ------------------------------
# model
# ------------------------------
# model training
params = {
    "objective": "regression",
    "verbosity": -1,
    "metric": "rmse",
    "interaction_constraints": [
        [0, 1, 2, 11, 12], 
        [3, 4], 
        [6, 10], 
        [5, 9], 
        [7, 8]
    ],
    "monotone_constraints": (1, 0, 1, -1, 1, 0, 1, 0, -1, 1, 1, -1, 1)
}
booster = lgb.train(
    params,
    train_set = train_dataset,
    valid_sets = (test_dataset,),
    num_boost_round = 100,
    early_stopping_rounds = 5,
    feval = mean_absolute_error,
)
# train predict
train_preds = booster.predict(X_train)
print(f"\nTrain R2 Score: {r2_score(y_train, train_preds)}")

# model cv

# ------------------------------
# model predict
# ------------------------------
test_preds = booster.predict(X_test)
idxs = booster.predict(X_test, pred_leaf = True)
shap_vals = booster.predict(X_test, pred_contrib = True)
print(f"\nTest R2 Score: {r2_score(y_test, test_preds)}")
print(f"\nidxs Shape: {idxs.shape}, \nidxs={idxs}")
print(f"\nshap_vals Shape: {shap_vals.shape}, \nshap_vals={shap_vals}")
print(f"\nShap Values of 0th Sample: {shap_vals[0]}")
print(f"\nPrediction of 0th using SHAP Values: {shap_vals[0].sum()}")
print(f"\nActual Prediction of 0th Sample: {test_preds[0]}")

# ------------------------------
# feature importance
# ------------------------------
gain_imp = booster.feature_importance(importance_type = "gain")
print(f"\ngain importance={gain_imp}")
split_imp = booster.feature_importance(importance_type = "split")
print(f"\nsplit importance={split_imp}")

# ------------------------------
# model plot
# ------------------------------
lgb.plot_importance(booster, figsize = (8, 6))
lgb.plot_metric(booster, figsize = (8, 6))
lgb.plot_metric(booster, metric = "rmse", figsize = (8, 6))
lgb.plot_split_value_histogram(booster, feature = "LSTAT", figsize = (8, 6))
lgb.plot_tree(booster, tree_index = 1, figsize = (20, 12))
plt.show()


# ------------------------------
# model save
# ------------------------------
model_file = "./models/lgb_regressor_boston.model"
if not os.path.exists(model_file):
    booster.save_model(model_file)

model_string = "./models/lgb_regressor_boston_booster.model"
if not os.path.exists(model_string):
    model_as_str = booster.model_to_string()
    with open(model_string, "w") as f:
        f.write(model_as_str)

# ------------------------------
# model load
# ------------------------------
if os.path.exists(model_file):
    loaded_booster = lgb.Booster(model_file = model_file)
    test_preds = loaded_booster.predict(X_test)
    print(f"\nTest R2 Score: {r2_score(y_test, test_preds)}")

if os.path.exists(model_string):
    model_str = open(model_string).read()
    loaded_booster_str = lgb.Booster(model_str = model_str)
    test_preds = loaded_booster_str.predict(X_test)
    print(f"\nTest R2 Score: {r2_score(y_test, test_preds)}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
