# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FeatureBuilding.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-06
# * Version     : 0.1.040614
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

from sklearn.preprocessing import PolynomialFeatures

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class FeatureBuilding:
    """
    特征生成
    """
    def __init__(self):
        pass

    def gen_polynomial_features(self, data, degree = 2, is_interaction_only = True, is_include_bias = True):
        """
        生成多项式特征
        Args:
            degree: 多项式阶数
            is_interaction_only: 是否只包含交互项
            is_include_bias: 是否包含偏差
        """
        pf = PolynomialFeatures(degree = degree, interaction_only = is_interaction_only, include_bias = is_include_bias)
        transformed_data = pf.fit_transform(data)
        return transformed_data




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
