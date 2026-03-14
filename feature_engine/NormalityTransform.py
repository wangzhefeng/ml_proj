# -*- coding: utf-8 -*-

# ***************************************************
# * File        : NormalityTransform.py
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

import numpy as np
import pandas as pd
from scipy.stats import skew

from sklearn.preprocessing import QuantileTransformer

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def SkewedFeatures(data, num_feat_idx, limit_value = 0.75) -> pd.Index:
    """
    筛选特征斜度高于某一阈值的特征

    Args:
        data (_type_): _description_
        num_feat_idx (_type_): _description_
        limit_value (float, optional): _description_. Defaults to 0.75.

    Returns:
        _type_: _description_
    """
    skewed_feat_values = data[num_feat_idx].apply(lambda x: skew(x.dropna()))
    skewed_feat_values = skewed_feat_values[np.abs(skewed_feat_values) > limit_value]

    return skewed_feat_values.index




# 测试代码 main 函数
def main():
    import pandas as pd
    from FeatureSplit import NumericCategoricalSplit

    df = pd.DataFrame({
        "a": range(10),
        "b": ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"],
        "c": [1, 2, 3, 1, 2, 2, 2, 3, 3, 1],
        "d": np.random.randn(10),
    })
    num_feat, num_feat_idx, cate_feat, cate_feat_idx = NumericCategoricalSplit(
        data = df, 
        limit_value = 4
    )
    print(num_feat, "\n")
    skewed_feat = df[num_feat_idx].apply(lambda x: skew(x.dropna()))
    print(skewed_feat)
    skew_df = skewed_feat[np.abs(skewed_feat) > 0.1]
    print(skew_df)
    print(skew_df.index)
    print(df[skew_df.index])

if __name__ == "__main__":
    main()
