# -*- coding: utf-8 -*-

# ***************************************************
# * File        : GridSearch_demo2.py
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

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import pandas as pd

# 导入数据
iris = datasets.load_iris()

# 定义超参搜索空间
parameters = {
    'kernel':('linear', 'rbf'), 
    'C':[1, 10]
}
# 初始化模型
svc = svm.SVC()

# 网格搜索
clf = GridSearchCV(
    estimator = svc,
    param_grid = parameters,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 5
)
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
