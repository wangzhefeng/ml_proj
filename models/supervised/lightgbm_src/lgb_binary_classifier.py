# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lgb_binary_classifier.py
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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from dataset.bclf_cancer import get_cancer

warnings.filterwarnings("ignore")
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data
# ------------------------------
# data
breast_cancer = get_cancer()
# data split
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target)
print(f"\nTrain/Test Sizes : {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")
# lgb data
train_dataset = lgb.Dataset(X_train, y_train, feature_name = breast_cancer.feature_names.tolist())
test_dataset = lgb.Dataset(X_test, y_test, feature_name = breast_cancer.feature_names.tolist())

# ------------------------------
# model
# ------------------------------
# model trianing
params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
}
evals_results = {}
booster = lgb.train(
    params,
    train_set = train_dataset,
    valid_sets = (test_dataset,),
    num_boost_round = 100,
    # early_stopping_rounds = 3,
    callbacks = [
        lgb.early_stopping(3),
        lgb.callback.print_evaluation(period = 3),
        lgb.record_evaluation(evals_results),
        lgb.reset_parameter(learning_rate = np.linspace(0.1, 1, 10).tolist())
    ],
)

train_preds = booster.predict(X_train)
train_preds = [1 if pred > 0.5 else 0 for pred in train_preds]
print(f"\nTrain Accuracy Score: {accuracy_score(y_train, train_preds)}")
print(f"Evaluation Results: {evals_results}")

# model cv
params_cv = {
    "objective": "binary",
    "verbosity": -1,
    "metrics": ["acc", "average_precision"],
    "eval_names": ["Validation Set"],
}
cv_output = lgb.cv(
    params_cv,
    train_set = train_dataset,
    num_boost_round = 10,
    folds = StratifiedShuffleSplit(n_splits = 3),
    verbose_eval = True,
    return_cvbooster = True,
)
for key, val in cv_output.items():
    print(f"\n{key} : {val}")

# ------------------------------
# model test
# ------------------------------
# model predict
test_preds = booster.predict(X_test)
test_preds = [1 if pred > 0.5 else 0 for pred in test_preds]
print(f"\nTest Accuracy Score: {accuracy_score(y_test, test_preds)}")

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
# final model training
# ------------------------------




# ------------------------------
# model save
# ------------------------------
model_file = "./models/lgb_bclf_breast.model"
if not os.path.exists(model_file):
    booster.save_model(model_file)

model_string = "./models/lgb_bclf_breast_booster.model"
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
    test_preds = [1 if pred > 0.5 else 0 for pred in test_preds]
    print(f"\nTest Accuracy Score: {accuracy_score(y_test, test_preds)}")

if os.path.exists(model_string):
    model_str = open(model_string).read()
    loaded_booster_str = lgb.Booster(model_str = model_str)
    test_preds = loaded_booster_str.predict(X_test)
    test_preds = [1 if pred > 0.5 else 0 for pred in test_preds]
    print(f"\nTest Accuracy Score: {accuracy_score(y_test, test_preds)}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
