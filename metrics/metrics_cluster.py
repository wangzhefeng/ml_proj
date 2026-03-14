# -*- coding: utf-8 -*-


# ***************************************************
# * File        : metrics_cluster.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-19
# * Version     : 0.1.031901
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np

from sklearn.metrics import make_scorer
from sklearn.metrics import rand_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import jaccard_score

from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class cluster_score:

    def __init__(self, labels_true, labels_pred, X = None, metric = "euclidean"):
        self.labels_true = labels_true
        self.labels_pred = labels_pred
        self.X = X
        self.metric = metric
    
    # ------------------------------
    # 外部指标
    # ------------------------------
    # rand index
    def rand_index(self):
        score = rand_score(self.labels_true, self.labels_pred)
        return score
    
    def RandIndex(self):
        # initialize variables
        a, b = 0, 0
        # compute variables
        for i in range(len(self.labels_true)):
            for j in range(i + 1, len(self.labels_true)):
                if self.labels_true[i] == self.labels_true[j] and self.labels_pred[i] == self.labels_pred[j]:
                    a += 1
                elif self.labels_true[i] != self.labels_true[j] and self.labels_pred[i] != self.labels_pred[j]:
                    b += 1
        # combinations
        from math import comb
        combinations = comb(len(self.labels_true), 2)
        # compute Rand Index
        rand_index = (a + b) / combinations
        return rand_index

    def adjusted_rand_index(self):
        score = adjusted_rand_score(self.labels_true, self.labels_pred)
        return score
    
    #  Fowlkes and Mallows Index(FM 指数)
    def fowlkes_mallows_index(self):
        score = fowlkes_mallows_score(self.labels_true, self.labels_pred)
        return score
    
    def FowlkesMallowsIndex(self):
        n = len(self.labels_true)
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if self.labels_true[i] == self.labels_true[j] and self.labels_pred[i] == self.labels_pred[j]:
                    tp += 1
                elif self.labels_true[i] != self.labels_true[j] and self.labels_pred[i] == self.labels_pred[j]:
                    fp += 1
                elif self.labels_true[i] == self.labels_true[j] and self.labels_pred[i] != self.labels_pred[j]:
                    fn += 1
                else:
                    tn += 1
        FM_score = tp / np.sqrt((tp + fp) * (tp + fn))
        return FM_score

    # Jaccard Coefficient
    def jaccard_coef(self):
        score = jaccard_score(self.labels_true, self.labels_pred)
        return score
    # ------------------------------
    # 内部指标
    # ------------------------------
    # Davies Bouldin Index
    def davies_bouldin_index(self):
        score = davies_bouldin_score(self.X, self.labels_pred)
        return score
    
    def DaviesBouldinIndex(self):
        def euclidean_distance(x, y):
            return np.sqrt(np.sum((x - y) ** 2))
        n_clusters = len(np.bincount(self.labels_pred))
        cluster_k = [self.X[self.labels_pred == k] for k in range(n_clusters)]
        centroids = [np.mean(k, axis = 0) for k in cluster_k]
        db_indices = []
        for i, k_i in enumerate(cluster_k):
            s_i = np.mean([np.linalg.norm(x - centroids[i]) for x in k_i])
            R = []
            for j, k_j in enumerate(cluster_k):
                if j != i:
                    s_j = np.mean([np.linalg.norm(x - centroids[j]) for x in k_j])
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    R.append((s_i + s_j) / dist)
            db_indices.append(np.max(R))
        return np.mean(db_indices)

    # Dunn Index
    def dunn_index(self):
        pass

    # Silhouette Coefficient
    def silhouette_coefficient(self):
        score = silhouette_score(self.X, self.labels_pred, self.metrics)
        return score

    def SilhouetteCoefficient(self):
        n_samples = len(self.X)
        cluster_labels = np.unique(self.labels_pred)
        n_clusters = len(cluster_labels)
        silhouette_vals = np.zeros(n_samples)
        for i in range(n_samples):
            a_i = np.mean([
                np.linalg.norm(self.X[i] - self.X[j]) 
                for j in range(n_samples) if self.labels_pred[j] == self.labels_pred[i] and j != i
            ])
            b_i = np.min([
                np.mean([np.linalg.norm(self.X[i] - self.X[j]) 
                for j in range(n_samples) if self.labels_pred[j] == k]) 
                for k in cluster_labels if k != self.labels_pred[i]
            ])
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
        silhouette_score = np.mean(silhouette_vals)
        return silhouette_score
    
    # Calinski Harabaz Index
    def calinski_harabaz_index(self):
        score = calinski_harabaz_score(self.X, self.labels_pred)
        return score

    def CalinskiHarabazIndex(self):
        n_samples, _ = self.X.shape
        n_labels = len(np.unique(self.labels_pred))
        if n_labels == 1:
            return np.nan
        mean = np.mean(self.X, axis=0)
        extra_disp, intra_disp = 0., 0.
        for k in range(n_labels):
            cluster_k = self.X[self.labels_pred == k]
            mean_k = np.mean(cluster_k, axis = 0)
            extra_disp += cluster_k.shape[0] * np.sum((mean_k - mean) ** 2)
            intra_disp += np.sum((cluster_k - mean_k) ** 2)
        chs = (extra_disp / (n_labels - 1)) / (intra_disp / (n_samples - n_labels))
        return chs
    # ------------------------------
    # 互信息
    # ------------------------------
    def mutual_info(self):
        score = mutual_info_score(self.labels_true, self.labels_pred)
        return score

    def adjust_mutual_info(self):
        score = adjusted_mutual_info_score(self.labels_true, self.labels_pred)
        return score

    def normalized_mutual_info(self):
        score = normalized_mutual_info_score(self.labels_true, self.labels_pred)
        return score
    # ------------------------------
    # 同质性和完整性
    # ------------------------------
    def homogeneity(self):
        score = homogeneity_score(self.labels_true, self.labels_pred)
        return score

    def completeness(self):
        score = completeness_score(self.labels_true, self.labels_pred)
        return score

    def v_measure(self):
        score = v_measure_score(self.labels_true, self.labels_pred)
        return score

    def homogeneity_completeness_v_measure(self):
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(self.labels_true, self.labels_pred)
        return homogeneity, completeness, v_measure
    # ------------------------------
    # 
    # ------------------------------
    def Contingency_Matrix(self):
        score = contingency_matrix(self.labels_true, self.labels_pred)
        return score

    def pair_confusion_matrix(self):
        pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
