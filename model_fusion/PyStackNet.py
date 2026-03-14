# -*- coding: utf-8 -*-

# ***************************************************
# * File        : PyStackNet.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101715
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





# 测试代码 main 函数
def main():
    from sklearn.ensemble import (
        RandomForestClassifier, 
        RandomForestRegressor, 
        ExtraTreesClassifier, 
        ExtraTreesRegressor, 
        GradientBoostingClassifier,
        GradientBoostingRegressor
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.decomposition import PCA

    from pystacknet.pystacknet import StackNetClassifier
    from pystacknet.pystacknet import StackNetRegressor

    models = [
        # First level
        [
            RandomForestClassifier(n_estimators = 100, criterion = "entropy", max_depth = 5, max_features = 0.5, random_state=1),
            ExtraTreesClassifier(n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, max_features=0.5, random_state=1),
            LogisticRegression(random_state=1)
        ],
        # Second level
        [
            RandomForestClassifier(n_estimators=200, criterion="entropy", max_depth=5, max_features=0.5, random_state=1)
        ]
    ]
    models = [ 
        [
            RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
            ExtraTreesRegressor(n_estimators=100, max_depth=5, max_features=0.5, random_state=1),
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, max_features=0.5, random_state=1),
            LogisticRegression(random_state=1),
            PCA(n_components=4, random_state=1)
        ],
        [
            RandomForestClassifier (n_estimators=200, criterion="entropy", max_depth=5, max_features=0.5, random_state=1)
        ]
    ]

    model = StackNetClassifier(
        models,
        metric = "auc",
        folds = 4,
        restacking = False,
        use_retraining = True,
        use_proba = True,
        random_state = 12345,
        n_jobs = 1,
        verbose = 1
    )
    model.fit(x, y)
    preds = model.predict_proba(x_test)

if __name__ == "__main__":
    main()
