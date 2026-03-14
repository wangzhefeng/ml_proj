# -*- coding: utf-8 -*-

# ***************************************************
# * File        : randomforest.py
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

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor


dt = DecisionTreeClassifier(
    max_depth = None,
    min_samples_split = 2,
    random_state = 0
)


rf = RandomForestClassifier(
    n_estimators = "warn",
    criterion = "gini",
    max_depth = None,
    min_samples_split = 2,
    min_samples_leaf = 1,
    min_weight_fraction_leaf = 0.0,
    max_features = "auto",
    max_leaf_nodes = None,
    min_impurity_decrease = 0.0,
    min_impurity_split = None,
    bootstrap = True,
    oob_score = False,
    n_jobs = None,
    random_state = None,
    verbose = 0,
    warm_start = False,
    class_weight = None
)


et = ExtraTreesClassifier(
    n_estimators = 'warn',
    criterion = 'gini',
    max_depth = None,
    min_samples_split = 2,
    min_samples_leaf = 1,
    min_weight_fraction_leaf = 0.0,
    max_features = 'auto',
    max_leaf_nodes = None,
    min_impurity_decrease = 0.0,
    min_impurity_split = None,
    bootstrap = False,
    oob_score = False,
    n_jobs = None,
    random_state = None,
    verbose = 0,
    warm_start = False,
    class_weight = None
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
