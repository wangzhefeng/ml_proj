# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-27
# * Version     : 0.1.022717
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys

import joblib
from fileinput import filename
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression







# 测试代码 main 函数
def main():
    print("Loading iris data set...")
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    print("Dataset loaded.")

    clf = LogisticRegression()
    pipe = Pipeline(
        [
            ("clf", clf)
        ]
    )
    print("Training model...")

    pipe.fit(X, y)
    print("Model trained.")

    filename_p = "/Users/zfwang/machinelearning/mlproj/utils/utils_deploy/deploy_fastapi/trained_models/IrisClassifier.pkl"
    print("Saving model in %s" % filename_p)

    joblib.dump(pipe, filename_p)
    print("Model saved!")

if __name__ == "__main__":
    main()
