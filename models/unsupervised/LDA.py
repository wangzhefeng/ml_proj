# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-19
# * Version     : 0.1.031920
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# data
iris = load_iris()

# 线性判别分析法，返回降维后的数据
# 参数 n_components 为降维后的维数
lda = LinearDiscriminantAnalysis(n_components = 2)
lda.fit_transform(iris.data, iris.target)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()