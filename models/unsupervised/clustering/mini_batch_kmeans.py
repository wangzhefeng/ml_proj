# -*- coding: utf-8 -*-


# ***************************************************
# * File        : mini_batch_kmeans.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-26
# * Version     : 0.1.112620
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import time

from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
batch_size = 45


# data
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples = 3000, centers = centers, cluster_std = 0.7)

# model
mbk = MiniBatchKMeans(
    init = "k-means++",
    n_clusters = 3,
    batch_size = batch_size,
    n_init = 10,
    max_no_improvement = 10,
    verbose = 0,
)
mbk.fit(X)








# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

