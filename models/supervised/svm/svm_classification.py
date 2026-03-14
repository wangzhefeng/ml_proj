# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
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
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# 数据
# ------------------------------
iris = datasets.load_iris()
X_train = iris["data"][:, (2, 3)]
y_train = (iris["target"] == 2).astype(np.float64)
print(X_train.shape)
print(y_train.shape)


# ------------------------------
# model
# ------------------------------
# SVC
svc_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svc", SVC(C = 1, kernel = "linear")),
))
svc_clf.fit(X_train, y_train)


# SVC：多项式核
poly_kernel_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(C = 5, kernel = "poly", degree = 3, coef0 = 1)),
))
poly_kernel_svm_clf.fit(X_train, y_train)


# LinearSVC
svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C = 1, loss = "hinge")),
))
svm_clf.fit(X_train, y_train)


# LinearSVC：多项式特征, 添加多项式特征, 使得数据线性可分
polynoimal_svm_clf = Pipeline((
    ("poly_features", PolynomialFeatures(degree = 3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C = 10, loss = "hinge")),
))
polynoimal_svm_clf.fit(X_train, y_train)


# SGDClassifier
m = 1
C = 1
sgd_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("sgd_svc", SGDClassifier(loss = "hinge", alpha = 1 / (m * C))),
))
sgd_clf.fit(X_train, y_train)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
