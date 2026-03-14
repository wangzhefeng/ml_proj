# -*- coding: utf-8 -*-


# ***************************************************
# * File        : IterLoadForDMatrix.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-06-26
# * Version     : 0.1.062622
# * Description : 在大规模数据集进行读取进行训练的过程中，
# *               迭代读取数据集是一个非常合适的选择，
# *               在 Pytorch 中支持迭代读取的方式
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
import xgboost as xgb


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class IterLoadForDMatrix(xgb.core.DataIter):
    """
    内存数据读取
    参考: https://xgboost.readthedocs.io/en/latest/python/examples/quantile_data_iterator.htm

    Args:
        xgb (_type_): _description_
    """
    def __init__(self, df = None, features = None, target = None, batch_size = 256 * 1024):
        self.df = df
        self.features = features
        self.target = target
        self.batch_size = batch_size
        self.batches = int(np.ceil(len(df) / self.batch_size))
        self.it = 0  # set iterator to 0
        super().__init__()

    def reset(self):
        """
        Reset the iterator
        """
        self.it = 0
    
    def next(self, input_data):
        """
        Yield next batch of data.

        Args:
            input_data (_type_): _description_
        """
        # Return 0 when there's no more batch.
        if self.it == self.batches:
            return 0
        
        a = self.it * self.batch_size
        b = min((self.it + 1) * self.batch_size, len(self.df))
        dt = pd.DataFrame(self.df.iloc[a:b])
        input_data(data = dt[self.features], label = dt[self.target])  # , weight = dt["weight"]
        self.it += 1
        return 1


class Iterator(xgb.Dataiter):
    """
    外部数据迭代读取
    参考: https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html

    Args:
        xgb (_type_): _description_
    """
    def __init__(self, svm_file_paths: List[str]):
        self._file_paths = svm_file_paths
        self._it = 0
        super().__init__(cache_prefix = os.path.join(".", "cache"))
    
    def next(self, input_data: Callable):
        if self._it == len(self._file_paths):
            # Return 0 to let XGBoost know this is the end of iteration
            return 0
        
        X, y = load_svmlight_file(self._file_paths[self._it])
        input_data(X, y)
        self._it += 1
        return 1

    def reset(self):
        """
        Reset the iterator to its begining
        """
        self._it = 0




# 测试代码 main 函数
def main():
    # 比较适合 GPU 训练
    Xy_train = IterLoadForDMatrix(train.loc[train_idx], FEATURES, "target")
    dtrain = xgb.DeviceQuantileDMatrix(Xy_train, max_bin = 256)

    # 比较适合 CPU 训练
    it = Iterator(["file_0.svm", "file_1.svm", "file_2.svm"])
    Xy = xgb.DMatrix(it)

    # Other tree methods including ``hist`` and ``gpu_hist`` also work, 
    # but has some caveats as noted in following sections.
    booster = xgb.train({"tree_method": "approx"}, Xy)

if __name__ == "__main__":
    main()
