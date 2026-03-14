# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FeatureText.py
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


def example_1():
    df = pd.DataFrame()
    df["feature"] = [
        "Apple_iPhone_6",
        "Apple_iPhone_6",
        "Apple_iPad_3",
        "Google_Pixel_3",
    ]
    df["feature_1st"] = df["feature"].apply(lambda x: x.split("_")[0])
    df["feature_2nd"] = df["feature"].apply(lambda x: x.split("_")[1])
    df["feature_3rd"] = df["feature"].apply(lambda x: x.split("_")[2])
    print(df)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
