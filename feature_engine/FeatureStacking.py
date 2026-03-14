# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FeatureStacking.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101716
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Stacking(object):

    def __init__(self, clf, train_x, train_y, test_x, clf_name, folds, label_split = None) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.clf = clf
        self.clf_name = clf_name        
        self.folds = folds
        self.label_split = label_split
        # 单模型循环后的数据
        self.train = np.zeros((self.train_x.shape[0], 1))
        self.test = np.zeros((self.test_x.shape[0], 1))
        self.test_pred = np.empty((self.folds, self.test_x.shape[0], 1))

    def sklearn_reg(self):
        # k-fold 交叉验证
        kf = KFold(n_splits = self.folds, shuffle = True, random_state = 0)
        cv_scores = []  # cv 分数
        for i, (train_index, test_index) in enumerate(kf.split(self.train_x, self.label_split)):
            # data split
            tr_x = self.train_x[train_index]
            tr_y = self.train_y[train_index]
            te_x = self.train_x[test_index]
            te_y = self.train_y[test_index]
            # model training
            self.clf.fit(tr_x, tr_y)
            pred = self.clf.predict(te_x).reshape(-1, 1)
            self.train[test_index] = pred
            self.test_pred[i, :] = self.clf.predict(self.test_x).reshape(-1, 1)
            cv_scores.append(mean_squared_error(te_y, pred))
            print(f"{self.clf_name} {i} fold score is:{cv_scores}")
        self.test[:] = self.test_pred.mean(axis = 0)
        # cv scores
        print(f"{self.clf_name}_score_list:{cv_scores}")
        print(f"{self.clf_name}_score_mean:{np.mean(cv_scores)}")
        return self.train.reshape(-1, 1), self.test.reshape(-1, 1)
    
    def xgb_reg(self):
        # k-fold 交叉验证
        kf = KFold(n_splits = self.folds, shuffle = True, random_state = 0)
        cv_scores = []  # cv 分数
        for i, (train_index, test_index) in enumerate(kf.split(self.train_x, self.label_split)):
            # data split
            tr_x = self.train_x[train_index]
            tr_y = self.train_y[train_index]
            te_x = self.train_x[test_index]
            te_y = self.train_y[test_index]
            # model training
            train_matrix = self.clf.DMatrix(tr_x, label = tr_y, missing = -1)
            test_matrix = self.clf.DMatrix(te_x, label = te_y, missing = -1)
            z = self.clf.DMatrix(self.test_x, label = te_y, missing = -1)
            params = {
                "booster": "gbtree",
                "eval_metric": "rmse",
                "gamma": 1,
                "min_child_weight": 1.5,
                "max_depth": 5,
                "lambda": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.7,
                "eta": 0.03,
                "tree_method": "exact",
                "seed": 2017,
                "nthread": 12,
            }
            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [
                (train_matrix, "train"), 
                (test_matrix, "eval")
            ]
            if test_matrix:
                model = self.clf.train(
                    params, 
                    train_matrix, 
                    num_boost_round = num_round, 
                    evals = watchlist, 
                    early_stopping_rounds = early_stopping_rounds
                )
                pred = model.predict(test_matrix, ntree_limit = model.best_ntree_limit).reshape(-1, 1)
                self.train[test_index] = pred
                self.test_pred[i, :] = model.predict(z, ntree_limit = model.best_ntree_limit).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pred))
                print(f"{self.clf_name} {i} fold score is:{cv_scores}")
        self.test[:] = self.test_pred.mean(axis = 0)
        # cv scores
        print(f"{self.clf_name}_score_list:{cv_scores}")
        print(f"{self.clf_name}_score_mean:{np.mean(cv_scores)}")
        return self.train.reshape(-1, 1), self.test.reshape(-1, 1)

    def lgb_reg(self):
        # k-fold 交叉验证
        kf = KFold(n_splits = self.folds, shuffle = True, random_state = 0)
        cv_scores = []  # cv 分数
        for i, (train_index, test_index) in enumerate(kf.split(self.train_x, self.label_split)):
            # data split
            tr_x = self.train_x[train_index]
            tr_y = self.train_y[train_index]
            te_x = self.train_x[test_index]
            te_y = self.train_y[test_index]
            # model training
            train_matrix = self.clf.Dataset(tr_x, label = tr_y)
            test_matrix = self.clf.Dataset(te_x, label = te_y)
            params = {
                "boosting_type": "gbdt",
                "objective": "regression_l2",
                "metric": "rmse",
                "min_child_weight": 1.5,
                "num_leaves": 2 ** 5,
                "lambda": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.7,
                "learning_rate": 0.03,
                "tree_method": "exact",
                "seed": 2017,
                "nthread": 12,
                "silent": True,
            }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = self.clf.train(
                    params, 
                    train_matrix, 
                    num_round = num_round, 
                    valid_sets = test_matrix, 
                    early_stopping_rounds = early_stopping_rounds
                )
                pred = model.predict(te_x, num_iteration = model.best_iteration).reshape(-1, 1)
                self.train[test_index] = pred
                self.test_pred[i, :] = model.predict(self.test_x, num_iteration = model.best_iteration).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pred))
                print(f"{self.clf_name} {i} fold score is:{cv_scores}")
        self.test[:] = self.test_pred.mean(axis = 0)
        # cv scores
        print(f"{self.clf_name}_score_list:{cv_scores}")
        print(f"{self.clf_name}_score_mean:{np.mean(cv_scores)}")
        return self.train.reshape(-1, 1), self.test.reshape(-1, 1)

    def sklearn_clf(self):
        # k-fold 交叉验证
        kf = KFold(n_splits = self.folds, shuffle = True, random_state = 0)
        cv_scores = []  # cv 分数
        for i, (train_index, test_index) in enumerate(kf.split(self.train_x, self.label_split)):
            # data split
            tr_x = self.train_x[train_index]
            tr_y = self.train_y[train_index]
            te_x = self.train_x[test_index]
            te_y = self.train_y[test_index]
            # model training
            self.clf.fit(tr_x, tr_y)
            pred = self.clf.predict_proba(te_x)
            self.train[test_index] = pred[:, 0].reshape(-1, 1)
            self.test_pred[i, :] = self.clf.predict_proba(self.test_x)[:, 0].reshape(-1, 1)
            cv_scores.append(log_loss(te_y, pred[:, 0].reshape(-1, 1)))
            print(f"{self.clf_name} {i} fold score is:{cv_scores}")
        self.test[:] = self.test_pred.mean(axis = 0)
        # cv scores
        print(f"{self.clf_name}_score_list:{cv_scores}")
        print(f"{self.clf_name}_score_mean:{np.mean(cv_scores)}")
        return self.train.reshape(-1, 1), self.test.reshape(-1, 1)
    
    def xgb_clf(self):
        # k-fold 交叉验证
        kf = KFold(n_splits = self.folds, shuffle = True, random_state = 0)
        cv_scores = []  # cv 分数
        for i, (train_index, test_index) in enumerate(kf.split(self.train_x, self.label_split)):
            # data split
            tr_x = self.train_x[train_index]
            tr_y = self.train_y[train_index]
            te_x = self.train_x[test_index]
            te_y = self.train_y[test_index]
            # model training
            train_matrix = self.clf.DMatrix(tr_x, label = tr_y, missing = -1)
            test_matrix = self.clf.DMatrix(te_x, label = te_y, missing = -1)
            z = self.clf.DMatrix(self.test_x, label = te_y, missing = -1)
            params = {
                "booster": "gbtree",
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "gamma": 1,
                "min_child_weight": 1.5,
                "max_depth": 5,
                "lambda": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.7,
                "eta": 0.03,
                "tree_method": "exact",
                "seed": 2017,
                "num_class": 2,
            }
            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, "train"), (test_matrix, "eval")]
            if test_matrix:
                model = self.clf.train(
                    params, train_matrix, num_boost_round = num_round, 
                    evals = watchlist, early_stopping_rounds = early_stopping_rounds
                )
                pred = model.predict(test_matrix, ntree_limit = model.best_ntree_limit).reshape(-1, 1)
                self.train[test_index] = pred[:, 0].reshape(-1, 1)
                self.test_pred[i, :] = model.predict(z, ntree_limit = model.best_ntree_limit)[:, 0].reshape(-1, 1)
                cv_scores.append(log_loss()(te_y, pred[:, 0].reshape(-1, 1)))
                print(f"{self.clf_name} {i} fold score is:{cv_scores}")
        self.test[:] = self.test_pred.mean(axis = 0)
        # cv scores
        print(f"{self.clf_name}_score_list:{cv_scores}")
        print(f"{self.clf_name}_score_mean:{np.mean(cv_scores)}")
        return self.train.reshape(-1, 1), self.test.reshape(-1, 1)

    def lgb_clf(self):
        # k-fold 交叉验证
        kf = KFold(n_splits = self.folds, shuffle = True, random_state = 0)
        cv_scores = []  # cv 分数
        for i, (train_index, test_index) in enumerate(kf.split(self.train_x, self.label_split)):
            # data split
            tr_x = self.train_x[train_index]
            tr_y = self.train_y[train_index]
            te_x = self.train_x[test_index]
            te_y = self.train_y[test_index]
            # model training
            train_matrix = self.clf.Dataset(tr_x, label = tr_y)
            test_matrix = self.clf.Dataset(te_x, label = te_y)
            params = {
                "boosting_type": "gbdt",
                "objective": "multiclass",
                "metric": "multi_logclass",
                "min_child_weight": 1.5,
                "num_leaves": 2 ** 5,
                "lambda_l2": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.7,
                "learning_rate": 0.03,
                "tree_method": "exact",
                "seed": 2017,
                "num_class": 2,
                "silent": True,
            }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = self.clf.train(
                    params, train_matrix, num_round, valid_sets = test_matrix, 
                    early_stopping_rounds = early_stopping_rounds
                )
                pred = model.predict(te_x, num_iteration = model.best_iteration)
                self.train[test_index] = pred[:, 0].reshape(-1, 1)
                self.test_pred[i, :] = model.predict(self.test_x, num_iteration = model.best_iteration)[:, 0].reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pred[:, 0].reshape(-1, 1)))
                print(f"{self.clf_name} {i} fold score is:{cv_scores}")
        self.test[:] = self.test_pred.mean(axis = 0)
        # cv scores
        print(f"{self.clf_name}_score_list:{cv_scores}")
        print(f"{self.clf_name}_score_mean:{np.mean(cv_scores)}")
        return self.train.reshape(-1, 1), self.test.reshape(-1, 1)




# 测试代码 main 函数
def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC, SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR, NuSVR, LinearSVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import SGDRegressor
    import xgboost
    import lightgbm
    """
    regression
    """
    def rf_reg(x_train, y_train, x_valid, kf, label_split = None):
        randomforest = RandomForestRegressor(n_estimators = 600, max_depth = 20, n_jobs = -1, random_state=2017, max_features = "auto", verbose = 1)
        stacker = Stacking(randomforest, x_train, y_train, x_valid, "rf_reg", kf, label_split=label_split)
        rf_train, rf_test = stacker.sklearn_reg()
        return rf_train, rf_test, "rf_reg"

    def ada_reg(x_train, y_train, x_valid, kf, label_split = None):
        adaboost = AdaBoostRegressor(n_estimators = 30, random_state = 2017, learning_rate = 0.01)
        stacker = Stacking(adaboost, x_train, y_train, x_valid, "ada_reg", kf, label_split=label_split)
        ada_train, ada_test = stacker.sklearn_reg()
        return ada_train, ada_test, "ada_reg"

    def gb_reg(x_train, y_train, x_valid, kf, label_split = None):
        gbdt = GradientBoostingRegressor(learning_rate = 0.04, n_estimators = 100, subsample = 0.8, random_state = 2017, max_depth = 5, verbose = 1)
        gbdt_train, gbdt_test = Stacking(gbdt, x_train, y_train, x_valid, "gb_reg", kf, label_split=label_split)
        return gbdt_train, gbdt_test, "gb_reg"

    def et_reg(x_train, y_train, x_valid, kf, label_split = None):
        extratree = ExtraTreesRegressor(n_estimators = 600, max_depth = 35, max_features = "auto", n_jobs = -1, random_state = 2017, verbose = 1)
        stacker = Stacking(extratree, x_train, y_train, x_valid, "et_reg", kf, label_split=label_split)
        et_train, et_test = stacker.sklearn_reg()
        return et_train, et_test, "et_reg"

    def lr_reg(x_train, y_train, x_valid, kf, label_split = None):
        lr_reg = LinearRegression(n_jobs = -1)
        stacker = Stacking(lr_reg, x_train, y_train, x_valid, "lr_reg", kf, label_split=label_split)
        lr_train, lr_test = stacker.sklearn_reg()
        return lr_train, lr_test, "lr_reg"

    def xgb_reg(x_train, y_train, x_valid, kf, label_split = None):
        stacker = Stacking(xgboost, x_train, y_train, x_valid, "xgb_reg", kf, label_split=label_split)
        xgb_train, xgb_test = stacker.xgb_reg()
        return xgb_train, xgb_test, "xgb_reg"

    def lgb_reg(x_train, y_train, x_valid, kf, label_split = None):
        stacker = Stacking(lightgbm, x_train, y_train, x_valid, "lgb_reg", kf, label_split=label_split)
        lgb_train, lgb_test = stacker.lgb_reg()
        return lgb_train, lgb_test, "lgb_reg"
    """
    classification
    """
    def rf_clf(x_train, y_train, x_valid, kf, label_split = None):
        randomforest = RandomForestClassifier(n_estimators = 1200, max_depth = 20, n_jobs = -1, random_state = 2017, max_features = "auto", verbose = 1)
        stacker = Stacking(randomforest, x_train, y_train, x_valid, "rf_clf", kf, label_split=label_split)
        rf_train, rf_test = stacker.sklearn_clf()
        return rf_train, rf_test, "rf_clf"

    def ada_clf(x_train, y_train, x_valid, kf, label_split = None):
        adaboost = AdaBoostClassifier(n_estimators = 50, random_state = 2017, learning_rate = 0.01)
        stacker = Stacking(adaboost, x_train, y_train, x_valid, "ada_clf", kf, label_split=label_split)
        ada_train, ada_test = stacker.sklearn_clf()
        return ada_train, ada_test, "ada_clf"

    def gb_clf(x_train, y_train, x_valid, kf, label_split = None):
        gbdt = GradientBoostingClassifier(learning_rate = 0.04, n_estimators = 100, subsample = 0.8, random_state = 2017, max_depth = 5, verbose = 1)
        stacker = Stacking(gbdt, x_train, y_train, x_valid, "gb_clf", kf, label_split=label_split)
        gbdt_train, gbdt_test = stacker.sklearn_clf()
        return gbdt_train, gbdt_test, "gb_clf"

    def et_clf(x_train, y_train, x_valid, kf, label_split = None):
        extratree = ExtraTreesClassifier(n_estimators = 1200, max_depth = 35, max_features = "auto", n_jobs = -1, random_state = 2017, verbose = 1)
        stacker = Stacking(extratree, x_train, y_train, x_valid, "et_clf", kf, label_split=label_split)
        et_train, et_test = stacker.sklearn_clf()
        return et_train, et_test, "et_clf"

    def lr_clf(x_train, y_train, x_valid, kf, label_split = None):
        logisticregression = LogisticRegression(n_jobs = -1, random_state = 2017, C = 0.1, max_iter = 200)
        stacker = Stacking(logisticregression, x_train, y_train, x_valid, "lr_clf", kf, label_split=label_split)
        lr_train, lr_test = stacker.sklearn_clf()
        return lr_train, lr_test, "lr_clf"

    def xgb_clf(x_train, y_train, x_valid, kf, label_split = None):
        stacker = Stacking(xgboost, x_train, y_train, x_valid, "xgb_clf", kf, label_split=label_split)
        xgb_train, xgb_test = stacker.xgb_clf()
        return xgb_train, xgb_test, "xgb_clf"

    def lgb_clf(x_train, y_train, x_valid, kf, label_split = None):
        stacker = Stacking(lightgbm, x_train, y_train, x_valid, "lgb_clf", kf, label_split=label_split)
        lgb_train, lgb_test = stacker.lgb_clf()
        return lgb_train, lgb_test, "lgb_clf"

    def gnb_clf(x_train, y_train, x_valid, kf, label_split = None):
        gnb = GaussianNB()
        stacker = Stacking(gnb, x_train, y_train, x_valid, "gnb_clf", kf, label_split=label_split)
        gnb_train, gnb_test = stacker.sklearn_clf()
        return gnb_train, gnb_test, "gnb_clf"

    def knn_clf(x_train, y_train, x_valid, kf, label_split = None):
        kneighbors = KNeighborsClassifier(kneighbors = 200, n_jobs = -1)
        stacker = Stacking(kneighbors, x_train, y_train, x_valid, "knn_clf", kf, label_split=label_split)
        knn_train, knn_test = stacker.sklearn_clf()
        return knn_train, knn_test, "knn_clf"

if __name__ == "__main__":
    main()
