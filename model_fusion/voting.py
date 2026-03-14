# -*- coding: utf-8 -*-

# ***************************************************
# * File        : voting.py
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

# model fusion: voting
from sklearn.ensemble import VotingClassifier
# base models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# data
X_train = None
y_train = None

# model fusion
ensemble_voting = VotingClassifier(
    estimators = [
        ('dtc', DecisionTreeClassifier(random_state = 42)),
        ('lr', LogisticRegression()),
        ('gnb', GaussianNB()),
        ('knn', KNeighborsClassifier()),
        ('svc', SVC())
    ],
    voting = 'hard',
    weights = None,
)
ensemble_voting.fit(X_train, y_train)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
