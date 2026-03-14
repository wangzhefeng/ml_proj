# -*- coding: utf-8 -*-


# ***************************************************
# * File        : birch.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-27
# * Version     : 0.1.112719
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from joblib import cpu_count
from itertools import cycle
from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.datasets import make_blobs
from sklearn.cluster import Birch


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data
# ------------------------------
xx = np.linspace(-22, 22, 10)
yy = np.linspace(-22, 22, 10)
xx, yy = np.meshgrid(xx, yy)
n_centers = np.hstack((np.ravel(xx)[:, np.newaxis], np.ravel(yy)[: np.newaxis]))
X, y = make_blobs(n_samples = 25000, centers = n_centers, random_state = 0)


colors_ = cycle(colors.cnames.keys())
fig = plt.figure(figsize = (12, 4))
fig.subplots_adjust(left = 0.04, right = 0.98, bottom = 0.1, top = 0.9)

birch_models = [
    Birch(threshold = 1.7, n_clusters = None),
    Birch(threshold = 1.7, n_clusters = 100),
]





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

