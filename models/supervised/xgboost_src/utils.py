# -*- coding: utf-8 -*-


# ***************************************************
# * File        : tuning.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-05
# * Version     : 0.1.040522
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data
# ------------------------------
def read_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    target = "Disbursed"
    IDcol = 'ID'
    predictors = [x for x in train.columns if x not in [target, IDcol]]

    return train, test, predictors, target


# ------------------------------
# XGBoost model and cross-validation
# ------------------------------
def modelFit(alg, dtrain, predictors, target, 
             scoring = 'auc', useTrainCV = True, 
             cv_folds = 5, early_stopping_rounds = 50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgb_train = xgb.DMatrix(data = dtrain[predictors].values, label = dtrain[target].values)
        cv_result = xgb.cv(
            params = xgb_param,
            dtrain = xgb_train,
            num_boost_round = alg.get_params()['n_estimators'],
            nfold = cv_folds,
            stratified = False,
            metrics = scoring,
            early_stopping_rounds = early_stopping_rounds,
            show_stdv = False
        )
        alg.set_params(n_estimators = cv_result.shape[0])

    alg.fit(dtrain[predictors], dtrain[target], eval_metric = scoring)
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    print("\nModel Report:")
    print("Accuracy: %.4f" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending = False)
    feat_imp.plot(kind = 'bar', title = "Feature Importances")
    plt.ylabel("Feature Importance Score")


# ------------------------------
# parameter tuning
# ------------------------------
def grid_search(train, predictors, target, param_xgb, param_grid, scoring, n_jobs, cv_method):
    grid_search = GridSearchCV(
        estimator = XGBClassifier(**param_xgb),
        param_grid = param_grid,
        scoring = scoring,
        n_jobs = n_jobs,
        iid = False,
        cv = cv_method
    )
    grid_search.fit(train[predictors], train[target])
    print(grid_search.cv_results_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    return grid_search




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
