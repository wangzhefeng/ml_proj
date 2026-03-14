# -*- coding: utf-8 -*-

# ***************************************************
# * File        : stacking.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101717
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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# data
X_train = None
y_train = None

# model fusion
base_learners = [
    ('l1', KNeighborsClassifier()),
    ('l2', DecisionTreeClassifier()),
    ('l3',SVC(gamma = 2, C = 1)),
]
model = StackingClassifier(
    estimators = base_learners, 
    final_estimator = LogisticRegression(),
    cv = 5
)
model.fit(X_train, y_train)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
