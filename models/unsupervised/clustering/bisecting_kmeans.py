# -*- coding: utf-8 -*-


# ***************************************************
# * File        : bisecting_kmeans.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-26
# * Version     : 0.1.112619
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import BisectingKMeans, KMeans


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
n_samples = 1000
n_centers = 2
random_state = 0
n_clusters_list = [2, 3, 4, 5]


# data
X, _ = make_blobs(n_samples = n_samples, centers = n_centers, random_state = random_state)
print(X.shape)

# model
fig, axs = plt.subplots(1, len(n_clusters_list), figsize = (20, 5))
axs = axs.T
for i, n_clusters in enumerate(n_clusters_list):
    # model fit
    bk = BisectingKMeans(
        n_clusters = n_clusters,
        random_state = 0
    )
    bk.fit(X)
    # result
    centers = bk.cluster_centers_
    labels = bk.labels_
    # result plot
    axs[i].scatter(X[:, 0], X[:, 1], s = 10, c = labels)
    axs[i].scatter(centers[:, 0], centers[:, 1], c = "r", s = 20)
    axs[i].set_title(f"{n_clusters} clusters")

for ax in axs.flat:
    ax.label_outer()
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

