# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lgb_pipeline.py
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
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
# from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# --------------------------------------------
# data
# --------------------------------------------
# 原始数据
df_train = pd.read_csv(
    "https://cdn.coggle.club/LightGBM/examples/binary_classification/binary.train", 
    header = None, 
    sep = "\t"
)
df_test = pd.read_csv(
    "https://cdn.coggle.club/LightGBM/examples/binary_classification/binary.test", 
    header = None, 
    sep = "\t"
)
W_train = pd.read_csv(
    "https://cdn.coggle.club/LightGBM/examples/binary_classification/binary.train.weight", 
    header = None
)[0]
W_test = pd.read_csv(
    "https://cdn.coggle.club/LightGBM/examples/binary_classification/binary.test.weight", 
    header = None
)[0]
y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis = 1)
X_test = df_test.drop(0, axis = 1)
num_train, num_feature = X_train.shape
print(num_train)
print(num_feature)

# 创建适用于 LightGBM 的数据
lgb_train = lgb.Dataset(X_train, y_train, weight = W_train, free_raw_data = False)
lgb_eval = lgb.Dataset(X_test, y_test, reference = lgb_train, weight = W_test, free_raw_data = False)

# --------------------------------------------
# model training
# --------------------------------------------
# model params
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}
# model
feature_name = ["feature_" + str(col) for col in range(num_feature)]
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = 10,
    valid_sets = lgb_train,
    feature_name = feature_name,
    categorical_feature = [21]
)

# --------------------------------------------
# 模型保存与加载
# --------------------------------------------
# txt
gbm.save_model("/saved_models/ligtgbm_models/model.txt")
print("Dumping model to JSON ...")

# json
model_json = gbm.dump_model()
with open("/saved_models/ligtgbm_models/model.json", "w+") as f:
    json.dump(model_json, f, indent = 4)

# --------------------------------------------
# 查看特征重要性
# --------------------------------------------
print("Feature names:", gbm.feature_name())
print("Feature importances:", list(gbm.feature_importance()))




# --------------------------------------------
# 重新训练
# --------------------------------------------
gbm = lgb.train(
    params, 
    lgb_train, 
    num_boost_round = 10, 
    init_model = "/saved_models/ligtgbm_models/model.txt", 
    valid_sets = lgb_eval, 
)
print("Finished 10 - 20 rounds with model file ...")

# --------------------------------------------
# 动态调整模型超参数
# --------------------------------------------
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = 10,
    init_model = gbm,
    learning_rates = lambda iter: 0.05 * (0.99 ** iter),
    valid_sets = lgb_eval
)
print("Finished 20 ~ 30 rounds with deacy learning rates...")


gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = 10,
    init_model = gbm,
    valid_sets = lgb_eval,
    callbacks = [lgb.reset_parameter(bagging_fraction = [0.7] * 5 + [0.6] * 5)]
)
print("Finised 30 ~ 40 rounds with changing bagging_fraction...")

# --------------------------------------------
# 自定义损失函数
# --------------------------------------------
def loglikelihood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess


def binary_error(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return "error", np.mean(labels != (preds > 0.5)), False


gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = 10,
    init_model = gbm,
    fobj = loglikelihood,
    feval = binary_error,
    valid_sets = lgb_eval
)
print("Finished 40 ~ 50 rounds with self-defined objective function and eval metric...")

# --------------------------------------------
# 调参方法
# --------------------------------------------
# ----------------------
# 人工调参
# ----------------------

# ----------------------
# 网格搜索
# ----------------------
# model
lg = lgb.LGBMClassifier(silent = False)
# hyperparameters
param_dist = {
    "max_depth": [4, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [300, 900, 1200],
    "n_estimators": [50, 100, 150]
}
# grid search cv
grid_search = GridSearchCV(lg, n_jobs = -1, param_grid = param_dist, cv = 5, scoring = "roc_auc", verbose = 5)
grid_search.fit(X_train, y_train)

# best hyper parameters
grid_search.best_estimator_
grid_search.best_score_

# ----------------------
# 贝叶斯优化
# ----------------------
def lgb_eval(max_depth, learning_rate, num_leaves, n_estimators):
    params = {"metrics": "auc"}
    params["max_depth"] = int(max(max_depth, 1))
    params["learning_rate"] = np.clip(0, 1, learning_rate)
    params["num_leaves"] = int(max(num_leaves, 1))
    params["n_estimators"] = int(max(n_estimators, 1))
    cv_result = lgb.cv(
        params, 
        df_train, 
        nfold = 5, 
        seed = 0, 
        verbose_eval = 200, 
        stratified = False
    )
    return 1.0 * np.array(cv_result["auc-mean"]).max()


# lgbBO = BayesianOptimization(
#     lgb_eval, 
#     {
#         "max_depth": (4, 8),
#         "learning_rate": (0.05, 0.2),
#         "num_leaves": (20, 1500),
#         "n_estimators": (5, 200)
#     },
#     random_state = 0
# )
# lgbBO.maximize(init_points = 5, n_iter = 50, acq = "ei")
# print(lgbBO.max)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
