# -*- coding: utf-8 -*-

# ***************************************************
# * File        : GridSearch_CV.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101719
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
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import (
    train_test_split,
    KFold,
    RepeatedKFold,
    cross_val_score,
    cross_validate,
    GridSearchCV,
    LeaveOneOut,
    LeavePOut,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    RepeatedStratifiedKFold,
    TimeSeriesSplit,
    GroupKFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    GroupShuffleSplit,
)
from sklearn.metrics import classification_report

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
# params
np.random.seed(27149)
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
n_splits = 5


# ------------------------------
# data
# ------------------------------
digits = datasets.load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)


tuned_parameters = [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
]
scores = ["precision", "recall"]

for score in scores:
    print("# ----------------------------------------------")
    print("# Tuning hyper-parameters for %s" % score)

    print("# ----------------------------------------------")
    print("# Best parameters set found on development set:")

    clf = GridSearchCV(
             estimator = svm.SVC(),
            param_grid = tuned_parameters,
            cv = 5,
            scoring="%s_macro" % score
    )
    clf.fit(X_train, y_train)
    print(clf.best_params_)

    print("# ----------------------------------------------")
    print("# Grid scores on development set:")
    print("# ----------------------------------------------")
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std, params))
    print("# ----------------------------------------------")
    print("# Detailed classification report:")
    print()
    print("# The model is trained on the full development set.")
    print("# The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))


def visualize_groups(classes, groups, name):
    fig, ax = plt.subplots()
    ax.scatter(
        range(len(groups)), [0.5] * len(groups), c = groups, 
        marker = '_', lw = 50, cmap = cmap_data
    )
    ax.scatter(
        range(len(groups)), [3.5] * len(groups), c = classes, 
        marker = '_', lw = 50, cmap = cmap_data
    )
    ax.set(
        ylim = [-1, 5], 
        yticks = [0.5, 3.5], 
        yticklabels = ['Data\ngroup', 'Data\nclass'],
        xlabel = 'Sample index'
    )


def plot_cv_indices(cv, X, y, groups, ax, n_splits, lw = 10):
    for i, (train, validate) in enumerate(cv.split(X = X, y = y, groups = groups)):
        indices = np.array([np.nan] * len(X))
        indices[validate] = 1
        indices[train] = 0
        print("Index: ", i)
        print("Validation Index: ", validate)
        print("Train Index: ", train)
        print("Indices: ", indices)
        print("=" * 80)
        ax.scatter(
            range(len(indices)), [i + 0.5] * len(indices), c = indices, 
            marker = '_', lw = lw, cmap = cmap_cv, vmin = -0.2, vmax = 1.20
        )
    ax.scatter(
        range(len(X)), [i + 1.5] * len(X), c = y,
        marker = '_', lw = lw, cmap = cmap_data
    )
    ax.scatter(
        range(len(X)), [i + 2.5] * len(X), c = groups, 
        marker = '_', lw = lw, cmap = cmap_data
    )
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(
        yticks = np.arange(n_splits + 2) + 0.5, 
        yticklabels = yticklabels,
        xlabel='Sample index', ylabel="CV iteration",
        ylim = [n_splits + 2.2, -0.2],
        xlim = [0, 100]
    )
    ax.set_title('{}'.format(type(cv).__name__), fontsize = 15)
    ax.legend(
        [Patch(color = cmap_cv(.8)), Patch(color = cmap_cv(.02))], 
        ['Testing set', 'Training set'], 
        loc = (1.02, .8)
    )
    return ax




# 测试代码 main 函数
def main():
    # data
    n_points = 100
    percentiles_classes = [0.1, 0.3, 0.6]
    X = np.random.randn(100, 10)
    y = np.hstack([[i] * int(100 * perc) for i, perc in enumerate(percentiles_classes)])
    groups = np.hstack([[i] * 10 for i in np.arange(10)])
    print(X.shape)
    print(y)
    print(groups)
    # test
    visualize_groups(y, groups, 'no_groups')
    # plot
    fig, ax = plt.subplots()
    cv = KFold(n_splits)
    plot_cv_indices(cv, X, y, groups, ax, n_splits)

if __name__ == "__main__":
    main()
