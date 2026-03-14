# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FeatureEncoding.py
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

import category_encoders as ce
import pandas as pd
from category_encoders import LeaveOneOutEncoder, TargetEncoder, WOEEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class CategoryFeatureEncoder(object):

    def __init__(self) -> None:
        pass

    def ValueCounts(self):
        """
        Examples:
            # data
            >>> df = pd.DataFrame({
                    '区域' : ['西安', '太原', '西安', '太原', '郑州', '太原'], 
                    '10月份销售' : ['0.477468', '0.195046', '0.015964', '0.259654', '0.856412', '0.259644'],
                    '9月份销售' : ['0.347705', '0.151220', '0.895599', '0236547', '0.569841', '0.254784']
                })
            # feature engine
            >>> df_counts = df['区域'].value_counts().reset_index()
            >>> df_counts.columns = ['区域', '区域频度统计']
            >>> print(df_counts)
            >>> df = df.merge(df_counts, on = ['区域'], how = 'left')
            >>> print(df)
        """
        pass


def oneHotEncoding(data, limit_value = 10):
    """
    One-Hot Encoding: pandas get_dummies
    """
    feature_cnt = data.shape[1]
    class_index = []
    class_df = pd.DataFrame()
    normal_index = []
    for i in range(feature_cnt):
        if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) < limit_value:
            class_index.append(i)
            class_df = pd.concat([class_df, pd.get_dummies(data.iloc[:, i], prefix = data.columns[i])], axis = 1)
        else:
            normal_index.append(i)
    data_update = pd.concat([data.iloc[:, normal_index], class_df], axis = 1)
    return data_update


def one_hot_encoder(feature):
    """
    One-Hot Encoding: sklearn.preprocessing.OneHotEncoder
    """
    enc = OneHotEncoder(categories = "auto")
    encoded_feature = enc.fit_transform(feature)
    return encoded_feature


def order_encoder(feature):
    """
    Ordinal Encoding: sklearn.preprocessing.OrdinalEncoder
    """
    enc = OrdinalEncoder()
    encoded_feats = enc.fit_transform(feature)
    return encoded_feats


def label_encoder(data):
    """
    Label Encoder: sklearn.preprocessing.LabelEncoder
    """
    le = LabelEncoder()
    for c in data.columns:
        if data.dtypes[c] == object:
            le.fit(data[c].astype(str))
            data[c] = le.transform(data[c].astype(str))
    return data




# 测试代码 main 函数
def main():
    # order
    le =  LabelEncoder()
    classes = [1, 2, 6, 4, 2]
    new_classes = le.fit_transform(classes)
    print(le.classes_)
    print(new_classes)

    le =  LabelEncoder()
    classes = ["paris", "paris", "tokyo", "amsterdam"]
    new_classes = le.fit_transform(classes)
    print(le.classes_)
    print(new_classes)
    
    enc = OrdinalEncoder()
    classes = [1, 2, 6, 4, 2]
    new_classes = enc.fit_transform(classes)
    print(enc.classes_)
    print(new_classes)

    # one-hot
    df = pd.DataFrame({
        "City": ["SF", "SF", "SF", "NYC", "NYC", "NYC", "Seattle", "Seattle", "Seattle"],
        "Rent": [3999, 4000, 4001, 3499, 3500, 3501, 2499, 2500, 2501]
    })
    df["Rent"].mean()
    one_hot_df = pd.get_dummies(df, prefix = "city")
    print(one_hot_df)

    # 虚拟编码
    df = pd.DataFrame({
        "City": ["SF", "SF", "SF", "NYC", "NYC", "NYC", "Seattle", "Seattle", "Seattle"],
        "Rent": [3999, 4000, 4001, 3499, 3500, 3501, 2499, 2500, 2501]
    })
    df["Rent"].mean()
    vir_df = pd.get_dummies(df, prefix = "city", drop_first = True)
    print(vir_df)

    # 效果编码
    df = pd.DataFrame({
        "City": ["SF", "SF", "SF", "NYC", "NYC", "NYC", "Seattle", "Seattle", "Seattle"],
        "Rent": [3999, 4000, 4001, 3499, 3500, 3501, 2499, 2500, 2501]
    })
    df["Rent"].mean()
    vir_df = pd.get_dummies(df, prefix = "city", drop_first = True)
    effect_df = vir_df[3:5, ["city_SF", "city_Seattle"]] = -1
    print(effect_df)

if __name__ == "__main__":
    main()
