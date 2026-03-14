# -*- coding: utf-8 -*-


# ***************************************************
# * File        : gbdt.py
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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data
# ------------------------------
n_samples = 1000
random_state = np.random.RandomState(13)
x1 = random_state.uniform(size = n_samples)
x2 = random_state.uniform(size = n_samples)
x3 = random_state.randint(0, 4, size = n_samples)
X = np.c_[x1, x2, x3]
X = X.astype(np.float32)
p = 1 / (1.0 + np.exp(-(np.sin(3 * x1) - 4 * x2 + x3)))
y = random_state.binomial(1, p, size = n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 9)
print(X_train.shape)
print(y_train.shape)
print()
print(X_test.shape)
print(y_test.shape)


# ===========================================================
# parameters config
# ===========================================================
params = {
    'n_estimators': 1200,
    'max_depth': 3,
    'subsample': 0.5,
    'learning_rate': 0.01,
    'min_samples_leaf': 1,
    'random_state': 3
}
n_estimators = params['n_estimators']
x = np.arange(n_estimators) + 1

# ===========================================================
# # Fit classifier with out-of-bag estimates
# ===========================================================
clf = GradientBoostingClassifier(**params)
clf.fit(X_train, y_train)
# OOB score, best iter
print(clf.oob_improvement_)

cumsum = -np.cumsum(clf.oob_improvement_)
oob_best_iter = x[np.argmin(cumsum)]

print(cumsum)
# ===========================================================
# Held Out
# ===========================================================
def heldout_score(clf, X_test, y_test):
    score = np.zeros((n_estimators,), dtype = np.float64)
    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        score[i] = clf.loss_(y_test, y_pred)

    return score

test_score = heldout_score(clf, X_test, y_test)

print(test_score)

# Held Out score, best iter
test_score -= test_score[0]
test_best_iter = x[np.argmin(test_score)]
print(test_score)

# ===========================================================
# K-Fold CV
# ===========================================================
def cv_estimate(n_splits = None):
    kf_cv = KFold(n_splits = n_splits)
    cv_clf = GradientBoostingClassifier(**params)
    val_scores = np.zeros((n_estimators,), dtype = np.float64)
    for train_index, valid_index in kf_cv.split(X_train, y_train):
        cv_clf.fit(X_train[train_index], y_train[train_index])
        val_scores += heldout_score(cv_clf, X_train[valid_index], y_train[valid_index])
    val_scores /= n_splits

    return val_scores

cv_score = cv_estimate(3)

print(cv_score)

# 5-Fold CV score, best iter
cv_score -= cv_score[0]
cv_best_iter = x[np.argmin(cv_score)]
print(cv_score)



# ==========================================================
# plot the loss of iterations
# ==========================================================
def plot_estimator(x, cumsum, test_score, cv_score, oob_best_iter, test_best_iter, cv_best_iter):
    # color brew for the three curves
    oob_color = list(map(lambda x: x / 256.0, (190, 174, 212)))
    test_color = list(map(lambda x: x / 256.0, (127, 201, 127)))
    cv_color = list(map(lambda x: x / 256.0, (253, 192, 134)))

    # plot curves and vertical lines for best iterations
    plt.plot(x, cumsum, label = 'OOB loss', color = oob_color)
    plt.plot(x, test_score, label = 'Test loss', color = test_color)
    plt.plot(x, cv_score, label = 'CV loss', color = cv_color)
    plt.axvline(x = oob_best_iter, color = oob_color)
    plt.axvline(x = test_best_iter, color = test_color)
    plt.axvline(x = cv_best_iter, color = cv_color)

    # add three vertical lines to xticks
    xticks = plt.xticks()
    xticks_pos = np.array(xticks[0].tolist() + [oob_best_iter, cv_best_iter, test_best_iter])
    xticks_label = np.array(list(map(lambda t: int(t), xticks[0])) + ['OOB', 'CV', 'Test'])
    ind = np.argsort(xticks_pos)
    xticks_pos = xticks_pos[ind]
    xticks_label = xticks_label[ind]
    plt.xticks(xticks_pos, xticks_label)

    plt.legend(loc = 'upper right')
    plt.ylabel('normalized loss')
    plt.xlabel('number of iterations')

    plt.show()

# plot_estimator(x, cumsum, test_score, cv_score, oob_best_iter, test_best_iter, cv_best_iter)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
