# -*- coding: utf-8 -*-

# ***************************************************
# * File        : RandomForestImportance.py
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def RandomForestClf(x_train, y_train, feature_labels, threshold = 0.15):
    # rf model
    rf_clf = RandomForestClassifier(
        n_estimators = 10000,
        random_state = 0,
        n_jobs = -1,
    )
    rf_clf.fit(x_train, y_train)
    # importance
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feature_labels[indices[f]], importances[indices[f]]))
    # features selection
    x_selected = x_train[:, importances > threshold]

    return x_selected


def RandomForestClf(x_train, y_train, feature_labels, threshold = 0.15):
    # rf model
    rf_clf = RandomForestRegressor(
        n_estimators = 10000,
        random_state = 0,
        n_jobs = -1,
    )
    rf_clf.fit(x_train, y_train)
    # importance
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feature_labels[indices[f]], importances[indices[f]]))
    # features selection
    x_selected = x_train[:, importances > threshold]

    return x_selected




# 测试代码 main 函数
def main():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # data
    data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    df = pd.read_csv(data_url, header = None)
    df.columns = [
        'Class label', 'Alcohol', 'Malic acid', 'Ash', 
        'Alcalinity of ash', 'Magnesium', 'Total phenols', 
        'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
        'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
    ]
    # data split
    x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
    feat_labels = df.columns[1:]

    # model
    rf_clf = RandomForestClassifier(
        n_estimators = 10000,
        random_state = 0,
        n_jobs = -1,
    )
    rf_clf.fit(x_train, y_train)

    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

    # features selection
    threshold = 0.15
    x_selected = x_train[:, importances > threshold]
    print(x_selected.shape)

if __name__ == "__main__":
    main()
