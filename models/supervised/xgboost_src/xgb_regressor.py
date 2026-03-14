# -*- coding: utf-8 -*-


# ***************************************************
# * File        : xgboostclassifier.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-05
# * Version     : 0.1.040511
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
_path = os.path.abspath(os.path.dirname(__file__))
if os.path.join(_path, "..") not in sys.path:
    sys.path.append(os.path.join(_path, ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb

from dataset.reg_boston import get_boston

import warnings
warnings.filterwarnings("ignore")
print(f"XGBoost Version: {xgb.__version__}")
print(f"Scikit-learn Version: {sklearn.__version__}")


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data
# ------------------------------
# data
boston = get_boston()
# data split
X_train, X_test, y_train, y_test = train_test_split(
    boston.data,
    boston.target,
    train_size = 0.9,
    random_state = 42,
)
print(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}, \
        y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape}")
# xgboost data
dmat_train = xgb.DMatrix(X_train, y_train, feature_names = boston.feature_names)
dmat_test = xgb.DMatrix(X_test, y_test, feature_names = boston.feature_names)

# ------------------------------
# model
# ------------------------------
# model training
params = {
    "objective": "reg:squarederror",
    "max_depth": 3,
    "eta": 1,
}
booster = xgb.train(
    params,
    dmat_train,
    evals = [
        (dmat_train, "train"), 
        (dmat_test, "test")
    ],
)

# model test
result = pd.DataFrame({
    "Actuals": y_test[:10],
    "Prediction": booster.predict(dmat_test, ntree_limit = 0, )[:10],
})
print(result)


# model interpret
shap_values = booster.predict(dmat_test, pred_contribs = True)
print(f"\nSHAP Values Size: {shap_values.shape}")
print(f"\nSample SHAP Values: {shap_values[0]}")
print(f"\nSumming SHAP Values for Prediction: {shap_values.sum(axis = 1)[:5]}")

booster.predict(dmat_test, pred_leaf = True)[:5]

shap_interactions = booster.predict(dmat_test, pred_interactions = True)
print(f"SHAP Interactions Size: {shap_interactions.shape}")


# model evaluate
# rmse
print(f"Train RMSE: {booster.eval(dmat_train)}")
print(f"Test  RMSE: {booster.eval(dmat_test)}")
# r2
print(f"Train R2 Score: {r2_score(y_train, booster.predict(dmat_train))}")
print(f"Test  R2 Score: {r2_score(y_test, booster.predict(dmat_test))}")



# ------------------------------
# model plot
# ------------------------------
# with plt.style.context("ggplot"):
#     fig = plt.figure(figsize = (9, 6))
#     ax = fig.add_subplot(111)
#     xgb.plotting.plot_importance(booster, ax = ax, height = 0.6, importance_type = "weight")
#     plt.show()


with plt.style.context("ggplot"):
    fig = plt.figure(figsize = (25, 10))
    ax = fig.add_subplot(111)
    xgb.plotting.plot_tree(booster, ax = ax, num_trees = 9)
    plt.show()






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
