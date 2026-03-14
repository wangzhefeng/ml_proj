# -*- coding: utf-8 -*-

# ***************************************************
# * File        : demo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101718
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# 数据读取
df = pd.read_csv('https://mirror.coggle.club/dataset/heart.csv')
X = df.drop(columns = ['output'])
y = df['output']

# 数据划分
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify = y)

# 模型训练与计算准确率
clf = RandomForestClassifier(random_state = 0)
clf.fit(x_train, y_train)
clf.score(x_test, y_test)

# ------------------------------
# 
# ------------------------------
# model selection
parameters = {
    'max_depth': [2, 4, 5, 6, 7],
    'min_samples_leaf': [1, 2, 3],
    'min_weight_fraction_leaf': [0, 0.1],
    'min_impurity_decrease': [0, 0.1, 0.2]
}

# Fitting 5 folds for each of 90 candidates, totalling 450 fits
clf = GridSearchCV(
    estimator = RandomForestClassifier(random_state = 0),
    param_grid = parameters, 
    scoring = "accuracy",
    n_jobs = -1,
    cv = 5,
    refit = True, 
    verbose = 1,
)
clf.fit(x_train, y_train)
test_score = clf.best_estimator_.score(x_test, y_test)

print('详细结果:\n', pd.DataFrame.from_dict(clf.cv_results_))
print('最佳分类器:\n', clf.best_estimator_)
print('最佳分数:\n', clf.best_score_)
print('最佳参数:\n', clf.best_params_)

# ------------------------------
# 
# ------------------------------
# data
x_train, y_train = None, None
x_test, y_test = None, None

# params
parameters = {
    'max_depth': [2,4,5,6,7],
    'min_samples_leaf': [1,2,3],
    'min_weight_fraction_leaf': [0, 0.1],
    'min_impurity_decrease': [0, 0.1, 0.2]
}

# model
clf = RandomizedSearchCV(
    estimator = RandomForestClassifier(random_state = 0),
    param_distributions = parameters, 
    n_iter = 10,
    scoring = "accuracy",
    cv = 5,
    refit = True, 
    n_jobs = -1,
    verbose = 1,
    random_state = 2023 
)

clf.fit(x_train, y_train)
clf.best_estimator_.score(x_test, y_test)

# 打印结果
print('详细结果:\n', pd.DataFrame.from_dict(clf.cv_results_))
print('最佳分类器:\n', clf.best_estimator_)
print('最佳分数:\n', clf.best_score_)
print('最佳参数:\n', clf.best_params_)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
