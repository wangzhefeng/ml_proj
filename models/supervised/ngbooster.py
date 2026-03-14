# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ngbooster.py
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

import pandas as pd
from ngboost.ngboost import NGBoost
from ngboost import NGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


"""
pip install --upgrade git+https://github.com/stanfordmlgroup/ngboost.git
"""


X, Y = load_boston(True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

ngb = NGBRegressor().fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# test Mean Seqared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
print("Test MSE", test_MSE)

# test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(Y_test.flatten()).mean()
print("Test NLL", test_NLL)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
