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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score
)
from sklearn import metrics


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data
# ------------------------------
def data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    target = "Disbursed"
    IDcol = 'ID'
    predictors = [x for x in train.columns if x not in [target, IDcol]]
    return train, test, predictors, target

# ------------------------------
# GBM model and cross-validation
# ------------------------------
def modelFit(alg, dtrain, predictors, target, performCV = True, printFeatureImportance = True, cv_method = 5):
    alg.fit(dtrain[predictors], dtrain[target])
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    if performCV:
        cv_score = cross_val_score(alg,
                                   dtrain[predictors], dtrain[target],
                                   cv = cv_method,
                                   scoring = 'roc_auc')

    print("\nModel Report")
    print("Accuracy: %.4f" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending = False)
        feat_imp.plot(kind = 'bar', title = 'Feature Importances')
        plt.ylabel('Feature Importance Score')



# ------------------------------
# parameter tuning
# ------------------------------
def grid_search(train, predictors, target, param_gbm, param_grid, scoring, cv_method, n_jobs, iid = False):
    grid_search = GridSearchCV(estimator = GradientBoostingClassifier(**param_gbm),
                               param_grid = param_grid,
                               scoring = scoring,
                               n_jobs = n_jobs,
                               iid = iid,
                               cv = cv_method)
    grid_search.fit(train[predictors], train[target])
    print(grid_search.cv_results_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    return grid_search

# -----------------------------------
# data
# -----------------------------------
train_path = "./data/GBM_XGBoost_data/Train_nyOWmfK.csv"
test_path = "./data/GBM_XGBoost_data/Test_bCtAN1w.csv"
train, test, predictors, target = data(train_path, test_path)

# -----------------------------------
# GBM 无调参模型
# -----------------------------------
gbm0 = GradientBoostingClassifier(random_state = 10)
cv_method = 5
modelFit(alg = gbm0,
         dtrain = train,
         predictors = predictors,
         target = target,
         performCV = True,
         printFeatureImportance = True,
         cv_method = cv_method)

# -----------------------------------
# GBM 基于默认的learning rate 调节树的数量
# n_estimators
# -----------------------------------
param_gbm_n_estimators = {
    'learning_rate': 0.1,
    'max_depth': 8,
    'min_samples_split': 500,
    'min_samples_leaf': 50,
    'max_features': 'sqrt',
    'subsample': 0.8,
    'random_state': 10
}
param_grid_n_estimators = {
    'n_estimators': range(20, 81, 10)
}
scoring = 'roc_auc'
cv_method = 5
n_jobs = 4

gbm_gird0 = grid_search(train = train,
                        predictors = predictors,
                        target = target,
                        param_gbm = param_gbm_n_estimators,
                        param_grid = param_grid_n_estimators,
                        scoring = scoring,
                        cv_method = cv_method,
                        n_jobs = n_jobs)


# -----------------------------------
# 调节基于树的模型
# max_depth, min_samples_split
# -----------------------------------
param_gbm_tree1 = {
    'learning_rate': 0.1,
    'n_estimators': 60,
    'min_samples_leaf': 50,
    'max_features': 'sqrt',
    'subsample': 0.8,
    'random_state': 10
}
param_grid_tree1 = {
    'max_depth': range(5, 16, 2),
    'min_samples_split': range(200, 1001, 200)
}
scoring = 'roc_auc'
cv_method = 5
n_jobs = 4

gbm_grid1 = grid_search(train = train,
                        predictors = predictors,
                        target = target,
                        param_gbm = param_gbm_tree1,
                        param_grid = param_grid_tree1,
                        scoring = scoring,
                        cv_method = cv_method,
                        n_jobs = n_jobs)

# -----------------------------------
# 调节基于树的模型
# min_samples_split, min_samples_leaf
# -----------------------------------
param_gbm_tree2 = {
    'learning_rate': 0.1,
    'n_estimators': 60,
    'max_depth': 9,
    'max_features': 'sqrt',
    'subsample': 0.8,
    'random_state': 10
}
param_grid_tree2 = {
    'min_samples_split': range(1000, 2000, 200),
    'min_samples_leaf': range(30, 71, 10)
}
scoring = 'roc_auc'
cv_method = 5
n_jobs = 4

gbm_grid2 = grid_search(train = train,
                        predictors = predictors,
                        target = target,
                        param_gbm = param_gbm_tree2,
                        param_grid = param_grid_tree2,
                        scoring = scoring,
                        cv_method = cv_method,
                        n_jobs = n_jobs)

modelFit(alg = gbm_grid2.best_estimator_,
         dtrain = train,
         predictors = predictors,
         target = target,
         performCV = True,
         printFeatureImportance = True,
         cv_method = cv_method)


# -----------------------------------
# 调节基于树的模型
# max_features
# -----------------------------------
param_gbm_tree3 = {
    'learning_rate': 0.1,
    'n_estimators': 60,
    'max_depth': 9,
    'min_samples_split': 1200,
    'min_samples_leaf': 60,
    'subsample': 0.8,
    'random_state': 10
}
param_grid_tree3 = {
    'max_features': range(7, 20, 2)
}
scoring = 'roc_auc'
cv_method = 5
n_jobs = 4
gbm_grid3 = grid_search(train = train,
                        predictors = predictors,
                        target = target,
                        param_gbm = param_gbm_tree3,
                        param_grid = param_grid_tree3,
                        scoring = scoring,
                        cv_method = cv_method,
                        n_jobs = n_jobs)

# -----------------------------------
# 调节基于树的模型
# subsample
# -----------------------------------
param_gbm_tree4 = {
    'learning_rate': 0.1,
    'n_estimators': 60,
    'max_depth': 9,
    'min_samples_split': 1200,
    'min_samples_leaf': 60,
    'max_features': 'sqrt',
    #'subsample': 0.8,
    'random_state': 10
}
param_grid_tree4 = {
    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
}
scoring = 'roc_auc'
cv_method = 5
n_jobs = 4
gbm_grid4 = grid_search(train = train,
                        predictors = predictors,
                        target = target,
                        param_gbm = param_gbm_tree4,
                        param_grid = param_grid_tree4,
                        scoring = scoring,
                        cv_method = cv_method,
                        n_jobs = n_jobs)

# -----------------------------------
# learning_rate = 0.1
# n_estimators = 60
# min_samples_split = 1200
# min_samples_leaf = 60
# max_depth = 9
# max_features = 7
# subsample = 0.85
# random_state = 10
# -----------------------------------
# 降低learning rate
# 增加n_estimators
# -----------------------------------
param_gbm_tune1 = {
    'learning_rate': 0.05,
    'n_estimators': 120,
    'max_depth': 9,
    'min_samples_split': 1200,
    'min_samples_leaf': 60,
    'max_features': 7,
    'subsample': 0.85,
    'random_state': 10
}
cv_method = 5
gbm_tune1 = GradientBoostingClassifier(**param_gbm_tune1)
modelFit(alg = gbm_tune1,
         dtrain = train,
         predictors = predictors,
         target = target,
         performCV = True,
         printFeatureImportance = True,
         cv_method = cv_method)



param_gbm_tune2 = {
    'learning_rate': 0.01,
    'n_estimators': 600,
    'max_depth': 9,
    'min_samples_split': 1200,
    'min_samples_leaf': 60,
    'max_features': 7,
    'subsample': 0.85,
    'random_state': 10
}
cv_method = 5
gbm_tune2 = GradientBoostingClassifier(**param_gbm_tune2)
modelFit(alg = gbm_tune2,
         dtrain = train,
         predictors = predictors,
         target = target,
         performCV = True,
         printFeatureImportance = True,
         cv_method = cv_method)



param_gbm_tune3 = {
    'learning_rate': 0.005,
    'n_estimators': 1200,
    'max_depth': 9,
    'min_samples_split': 1200,
    'min_samples_leaf': 60,
    'max_features': 7,
    'subsample': 0.85,
    'random_state': 10
}
cv_method = 5
gbm_tune3 = GradientBoostingClassifier(**param_gbm_tune3)
modelFit(alg = gbm_tune3,
         dtrain = train,
         predictors = predictors,
         target = target,
         performCV = True,
         printFeatureImportance = True,
         cv_method = cv_method)



param_gbm_tune4 = {
    'learning_rate': 0.005,
    'n_estimators': 1500,
    'max_depth': 9,
    'min_samples_split': 1200,
    'min_samples_leaf': 60,
    'max_features': 7,
    'subsample': 0.85,
    'random_state': 10
}
cv_method = 5
gbm_tune4 = GradientBoostingClassifier(**param_gbm_tune4)
modelFit(alg = gbm_tune4,
         dtrain = train,
         predictors = predictors,
         target = target,
         performCV = True,
         printFeatureImportance = True,
         cv_method = cv_method)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
