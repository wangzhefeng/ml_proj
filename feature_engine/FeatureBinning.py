# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FeatureBinning.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-06
# * Version     : 0.1.040612
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

from sklearn.preprocessing import (
    Binarizer,
    KBinsDiscretizer,
    LabelBinarizer,
    MultiLabelBinarizer,
    binarize,
    label_binarize,
    FunctionTransformer,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def binarization(feature: pd.Series, threshold = 0.0, is_copy = True):
    """
    DONE
    数值特征二值化
    """
    transfer = Binarizer(threshold = threshold, copy = is_copy)
    transformed_data = transfer.fit_transform(
        np.array(feature).reshape(-1, 1)
    )

    return transformed_data.reshape(1, -1)[0]


def kbins(feature: pd.Series, n_bins, 
          encode = "ordinal", strategy = "quantile"):
    """
    分箱离散化

    Args:
        data (_type_): _description_
        n_bins (_type_): _description_
        encoder (str, optional): _description_. Defaults to "ordinal".
            - "ordinal"
            - "onehot"
            - "onehot-dense"
        strategy (str, optional): _description_. Defaults to "quantile".
            - "uniform"
            - "quantile"
            - "kmeans"
    
    Returns:
        _type_: _description_
    """
    transfer = KBinsDiscretizer(n_bins = n_bins, encode = encode, strategy = strategy)
    transformed_data = transfer.fit_transform(
        np.array(feature).reshape(-1, 1)
    )

    return transformed_data




# 测试代码 main 函数
def main():
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({
        "a": range(10),
        "b": ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"],
        "c": [1, 2, 3, 1, 2, 2, 2, 3, 3, 1],
        "d": np.random.randn(10),
    })
    print(df)
    binarization_data = binarization(
        feature = df["a"],
        threshold = 1.0,
    )
    print(binarization_data)

    k_bins_data = kbins(
        feature = df["d"],
        n_bins = 5,
    )
    print(k_bins_data)

if __name__ == "__main__":
    main()
