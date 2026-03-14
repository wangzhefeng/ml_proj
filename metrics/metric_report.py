# -*- coding: utf-8 -*-


# ***************************************************
# * File        : metric_report.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-06
# * Version     : 0.1.040600
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
import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    average_precision_score,
    recall_score,
    precision_recall_curve,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
    auc,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def test(y_true, y_pred):
    np.set_printoptions(precision = 2)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    print(cm)
    print(TN, FP, FN, TP)


def PrecisionRecall(y_true, y_pred):
    average_precision = average_precision_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    recall_macro = recall_score(y_true, y_pred, average = 'macro')
    recall_micro = recall_score(y_true, y_pred, average = 'micro')
    recall_weighted = recall_score(y_true, y_pred, average = 'weighted')
    recall_binary = recall_score(y_true, y_pred, average = 'binary')
    recall_samples = recall_score(y_true, y_pred, average = 'samples')


def precision_recall_curve_report(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    step_kwargs = {'step': 'post'}
    plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')
    plt.fill_between(recall, precision, alpha = 0.2, color = 'b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Two-Class Precision-Call Curve: AP = {0:0.2f}'.format(average_precision))


def ROC_plot(FPR, TPR, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(FPR, TPR, color = 'darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Postive Rate(FPR)')
    plt.ylabel('True Postive Rate(TPR)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc = 'lower right')
    plt.show()


def confusion_matrix_report(y_true, y_pred, 
                            classes, 
                            normalize = False, 
                            title = "Confusion Matrix", 
                            cmap = plt.cm.Blues):
    """
    混系矩阵报告和可视化
    """
    # 混淆矩阵
    np.set_printoptions(precision = 2)
    cm = confusion_matrix(y_true, y_pred)

    # 正规化混淆矩阵
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print('正规化后的混淆矩阵: ')
    else:
        print('没有进行正规化的混淆矩阵: ')
    
    # 混淆矩阵报告
    print(cm)

    # 混淆矩阵可视化
    plt.figure()
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.xticks(np.arange(len(classes)), classes, rotation = 45)
    plt.yticks(np.arange(len(classes)), classes)
    fmt = '0.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt), 
            horizontalalignment = 'center',
            color = 'white' if cm[i, j] > thresh else 'black'
        )
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()


def param_cvsearch_report(search_instance, n_top = 3):
    """
    模型交叉验证调参结果报告

    Args:
        search_instance (_type_): GridSearchCV, RandomizedSearchCV 实例
        n_top (int, optional): _description_. Defaults to 3.
    """
    results = search_instance.cv_results_
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:0.3f})".format(
                results["mean_test_score"][candidate],
                results["std_test_score"][candidate])
            )
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
