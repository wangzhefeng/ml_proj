# -*- coding: utf-8 -*-


# ***************************************************
# * File        : dbscan.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-23
# * Version     : 0.1.112322
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data
# ------------------------------
centers = [
    [1, 1],
    [-1, -1],
    [1, -1]
]
X, labels_true = make_blobs(
    n_samples = 750,
    centers = centers,
    cluster_std = 0.4,
    random_state = 0
)
X = StandardScaler().fit_transform(X)
# plt.plot(X[:, 0], X[:, 1], "o")
# plt.show()


def DBSCAN_clustering(X):
    """
    DBSCAN clustering

    Args:
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    db = DBSCAN(eps = 0.3, min_samples = 10).fit(X)
    # 聚类标签
    labels_pred = db.labels_
    # print(labels_pred)
    # 核心点索引
    core_sample_indices = db.core_sample_indices_
    # 核心点样本索引
    core_samples_mask = np.zeros_like(labels_pred, dtype = bool)
    core_samples_mask[core_sample_indices] = True
    # print(core_samples_mask)
    # 聚类的类别数(非离群点)
    n_clusters_ = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
    # print(n_clusters_)
    # 离群点数量
    n_noise_ = list(labels_pred).count(-1)
    # print(n_noise_)

    return labels_pred, core_samples_mask, n_clusters_


def DBSCAN_2D_plot(labels: np.ndarray, core_samples_mask: np.ndarray, n_clusters_: int):
    # 聚类的类别数(包含离群点)
    unique_labels = set(labels)
    print(unique_labels)
    # 4 组颜色
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    print(colors)
    for k, col in zip(unique_labels, colors):
        # 离群点数据颜色设置为黑色
        if k == -1:
            col = [0, 0, 0, 1]
        # 绘图
        class_member_mask = labels == k
        # 核心点
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0], 
            xy[:, 1], 
            "o", 
            markerfacecolor = tuple(col), 
            markeredgecolor = "k", 
            markersize = 14
        )
        # 非核心点
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0], 
            xy[:, 1], 
            "o", 
            markerfacecolor = tuple(col), 
            markeredgecolor = "k", 
            markersize = 6
        )
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()



# 测试代码 main 函数
def main():
    # ------------------------------
    # clustering
    # ------------------------------
    labels_pred, core_samples_mask, n_clusters_ = DBSCAN_clustering(X)
    # ------------------------------
    # metrics
    # ------------------------------
    # Homogeneity
    homogeneity = metrics.homogeneity_score(labels_true, labels_pred)
    print(homogeneity)
    # completeness
    completeness = metrics.completeness_score(labels_true, labels_pred)
    print(completeness)
    # V-measure
    v_measure = metrics.v_measure_score(labels_true, labels_pred)
    print(v_measure)
    # adjust rank index
    adjust_rank_index = metrics.adjusted_rand_score(labels_true, labels_pred)
    print(adjust_rank_index)
    # adjust mutual information
    adjust_mutual_info = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    print(adjust_mutual_info)
    # silhouette coefficient
    silhouette_coefficient = metrics.silhouette_score(X, labels_pred)
    print(silhouette_coefficient)
    # ------------------------------
    # clustering result plot
    # ------------------------------
    DBSCAN_2D_plot(labels_pred, core_samples_mask, n_clusters_)

if __name__ == "__main__":
    main()

