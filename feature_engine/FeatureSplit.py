# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FeatureSplit.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101716
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

import pandas as pd

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def NumericCategoricalSplit(data: pd.DataFrame, limit_value: int = 0):
    """
    数据集数值型、类别型特征分割

    Args:
        data (pd.DataFrame): _description_
        limit_value (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    # 特征分割
    num_feat_idx = []  # 数值特征名称
    cate_feat_idx = []  # 类别特征名称
    for i in data.columns:
        if (data[i].dtypes != "object") & (len(set(data[i])) >= limit_value):
            num_feat_idx.append(i)
        else:
            cate_feat_idx.append(i)
    # 数值特征
    num_feat_df = data[num_feat_idx]
    # 类别特征
    cate_feat_df = data[cate_feat_idx]

    return num_feat_df, num_feat_idx, cate_feat_df, cate_feat_idx




# 测试代码 main 函数
def main():
    import numpy as np

    df = pd.DataFrame({
        "a": range(10),
        "b": ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"],
        "c": [1, 2, 3, 1, 2, 2, 2, 3, 3, 1],
        "d": np.random.randn(10),
    })
    print(df)
    print(f"df.dtypes: {df.dtypes}")
    num_feat, num_feat_index, cate_feat, cate_feat_index = NumericCategoricalSplit(
        data = df, 
        limit_value = 4
    )
    print(num_feat)
    print(num_feat_index)
    print(cate_feat)
    print(cate_feat_index)
    
if __name__ == "__main__":
    main()
