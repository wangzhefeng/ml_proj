# -*- coding: utf-8 -*-

# ***************************************************
# * File        : RandomSearchCV.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101718
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
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, datasets
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# 导入数据
iris = datasets.load_iris()

# 定义超参搜索空间
distributions = {'kernel':['linear', 'rbf'], 'C':uniform(loc=1, scale=9)}

# 初始化模型
svc = svm.SVC()

# 网格搜索
clf = RandomizedSearchCV(estimator = svc,
                         param_distributions = distributions,
                         n_iter = 4,
                         scoring = 'accuracy',
                         cv = 5,
                         n_jobs = -1,
                         random_state = 2021)
clf.fit(iris.data, iris.target)

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