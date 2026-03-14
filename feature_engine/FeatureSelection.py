# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FeatureSelection.py
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
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import (
    PCA,
    IncrementalPCA,
    KernelPCA,
    TruncatedSVD,
    SparsePCA,
    FactorAnalysis,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectionKBest,
    chi2,
    RFE,
    SelectFromModel,
    SelectPercentile
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from minepy import MINE

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class pca:
    def __init__(self,
                 X, n_components,
                 whiten = False,
                 copy = True,
                 svd_solver = "auto",
                 tol = 0.0,
                 iterated_power = "auto",
                 random_state = None,
                 batch_size = None):
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
        pca = PCA(n_components = self.n_components,
                  copy = self.copy,
                  whiten = self.whiten,
                  svd_solver = self.svd_solver,
                  tol = self.tol,
                  iterated_power = self.iterated_power,
                  random_state = self.random_state)
        pca.fit_transform(self.X)

        return pca

    def incremental_pca(self):
        pca = IncrementalPCA(n_components = self.n_components,
                             whiten = self.whiten,
                             copy = self.copy,
                             batch_size = self.batch_size)
        # pca.partial_fit(self.X)
        pca.fit_transform(self.X)

        return pca


class lda:
    def __init__(self, X, n_components):
        self.X = X
        self.n_components = n_components

    def lda(self):
        lda = LinearDiscriminantAnalysis(n_components = self.n_components)
        lda.fit_transform(self.X)

        return lda


def nan_feature_remove(data, rate_base = 0.4):
    """
    针对每一列 feature 统计 nan 的个数
    个数大于全量样本的 rate_base 的认为是异常 feature, 进行剔除
    """
    all_cnt = data.shape[0]
    feature_cnt = data.shape[1]
    available_index = []
    for i in range(feature_cnt):
        rate = np.isnan(np.array(data.iloc[:, i])).sum() / all_cnt
        if rate <= rate_base:
            available_index.append(i)
    data_available = data.iloc[:, available_index]
    return data_available, available_index


def low_variance_feature_remove(data, rate_base = 0.0):
    """
    对样本数据集中方差小于某一阈值的特征进行剔除
    """
    sel = VarianceThreshold(threshold = rate_base)
    data_available = sel.fit_transform(data)

    return data_available


def col_filter(mtx_train, y_train, mtx_test, func=chi2, percentile=90):
    feature_select = SelectPercentile(func, percentile=percentile)
    feature_select.fit(mtx_train, y_train)
    mtx_train = feature_select.transform(mtx_train)
    mtx_test = feature_select.transform(mtx_test)
    
    return mtx_train, mtx_test


def model_based_feature_selection(data, target, model = "tree", n_estimators = 50):
    if model == "tree":
        clf = ExtraTreesClassifier(n_estimators = n_estimators).fit(data, target)
        model = SelectFromModel(clf, prefit=True)
        data_available = model.transform(data)
        return data_available
    elif model == "svm":
        clf = LinearSVC(C = 0.01, penalty = "l1", dual = False).fit(data, target)
        model = SelectFromModel(clf, prefit=True)
        data_available = model.transform(data)
        return data_available
    elif model == "lr":
        clf = LogisticRegression(C = 0.01, penalty = "l1", dual = False).fit(data, target)
        model = SelectFromModel(clf, prefit=True)
        data_available = model.transform(data)
        return data_available
    elif model == "lasso":
        clf = ""
    else:
        print("Error model, Please choose one of 'tree', 'svm' or 'lr'!")



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
