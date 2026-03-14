# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lgb_clf.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-31
# * Version     : 1.0.073113
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def kfold_lightgbm(df, num_folds, stratified = False, debug = False):
    """
    LightGBM with KFold or Stratified KFold
    Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
    :param df:
    :param num_folds:
    :param stratified:
    :param debug:
    :return:
    """
    # Divide in training / validation and testing data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 1001)
    else:
        folds = KFold(n_splits = num_folds, shuffle = True, random_state = 1001)

    # Create array and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_PREV', 'index']]
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread = 4,
            n_estimators = 10000,
            learning_rate = 0.02,
            num_leaves = 34,
            colsample_bytree = 0.9497036,
            subsample = 0.8715623,
            max_depth = 8,
            reg_alpha = 0.041545473,
            reg_lambda = 0.0735294,
            min_split_gain = 0.0222415,
            min_child_weight = 39.3259775,
            silent = -1,
            verbose = -1,
        )
        clf.fit(
            train_x, train_y,
            eval_set = [(train_x, train_y), (valid_x, valid_y)],
            eval_metric = 'auc',
            verbose = 200,
            early_stopping_rounds = 200,
        )
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration = clf.best_iteration_)[:, -1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration = clf.best_iteration_)[:, -1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = feats
        fold_importance_df['importance'] = clf.feature_importances_
        fold_importance_df['fold'] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df])
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
    print("Full AUC score %.6f" % roc_auc_score(train_df['TARGET'], oof_preds))

    # Display feature importance
    display_importances(feature_importance_df)

    return feature_importance_df


def display_importances(feature_importance_df_):
    """
    Display / plot feature importance
    """
    cols = feature_importance_df_[['feature', 'importance']] \
               .groupby('feature') \
               .mean() \
               .sort_values(by = 'importance', ascending = False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize = (8, 10))
    sns.barplot(
        x = 'importance', 
        y = 'feature', 
        data = best_features.sort_values(by = 'importance', ascending = False)
    )
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig("lgbm_importances.png");



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
