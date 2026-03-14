# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-19
# * Version     : 0.1.031919
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def loadData(fileName, delim = "\t"):
    data = pd.read_csv(fileName, sep = delim, header = None)
    return np.mat(data)


def PCA(dataMat, topNfeat = 9999999):
    meanVals = np.mean(dataMat, axis = 0)
    meanRemoved = dataMat - meanVals 					# 标准化
    covMat = np.cov(meanRemoved, rowvar = 0)		   	# 计算样本协方差矩阵
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))	# 对样本协方差矩阵进行特征分解, 得到特征向量和对应的特征值
    eigValInd = np.argsort(eigVals)				   		# 对特征值进行排序
    eigValInd = eigValInd[:-(topNfeat + 1):-1]	   		# 取最大的topNfeat个特征向量对应的index序号
    redEigVects = eigVects[:, eigValInd]		   		# 根据取到的特征值对特征向量进行排序
    lowDDataMat = meanRemoved * redEigVects				# 降维之后的数据集
    reconMat = (lowDDataMat * redEigVects.T) + meanVals # 新的数据空间

    return lowDDataMat, reconMat


def show_picture(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90,c='green')
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()




# 测试代码 main 函数
def main():
    data = loadData(fileName = "PCA.txt", delim = "\t")
    lowDDataMat, reconMat = PCA(data, 1)
    show_picture(data, reconMat)

if __name__ == "__main__":
    main()
