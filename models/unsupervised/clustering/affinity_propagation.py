# -*- coding: utf-8 -*-


# ***************************************************
# * File        : affinity_propagation.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-26
# * Version     : 0.1.112623
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.cluster import AffinityPropagation


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data
# ------------------------------
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples = 300,
    centers = centers,
    cluster_std = 0.5,
    random_state = 0,
)

# ------------------------------
# model
# ------------------------------
af = AffinityPropagation(
    preference = -50,
    random_state = 0,
)
af.fit(X)

# ------------------------------
# model result
# ------------------------------
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(cluster_centers_indices)
print(cluster_centers_indices)
print(labels)
print(n_clusters_)

# ------------------------------
# model performance
# ------------------------------
homogeneity = metrics.homogeneity_score(labels_true, labels)
completeness = metrics.completeness_score(labels_true, labels)
v_measure = metrics.v_measure_score(labels_true, labels)
adjusted_rand_index = metrics.adjusted_rand_score(labels_true, labels)
adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels_true, labels)
silhouette_score = metrics.silhouette_score(X, labels, metric = "sqeuclidean")

# ------------------------------
# model result plot 
# ------------------------------
plt.close("all")
plt.figure(1)
plt.clf()

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
for k, col in zip(range(n_clusters_), colors):
    # 类型为 k 的样本布尔索引
    class_members = labels == k
    X_x = X[class_members, 0]
    X_y = X[class_members, 1]
    # 类型为 k 的簇中心点
    cluster_center = X[cluster_centers_indices[k]]
    cluster_center_x = cluster_center[0]
    cluster_center_y = cluster_center[1]
    # 样本图
    plt.plot(X_x, X_y, col + ".")
    plt.plot(
        cluster_center_x, 
        cluster_center_y, 
        "o", 
        markerfacecolor = col, 
        markeredgecolor = "k", 
        markersize = 16
    )
    for x in X[class_members]:
        plt.plot(
            [cluster_center_x, x[0]], 
            [cluster_center_y, x[1]], 
            col
        )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

