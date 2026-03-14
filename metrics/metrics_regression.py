# -*- coding: utf-8 -*-


# ***************************************************
# * File        : metrics_regression.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-19
# * Version     : 0.1.031903
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.metrics import explained_variance_score
# MAE
from sklearn.metrics import mean_absolute_error
# MSE
from sklearn.metrics import mean_squared_error
# MSLE
from sklearn.metrics import mean_squared_log_error
# MAE
from sklearn.metrics import median_absolute_error
# R2
from sklearn.metrics import r2_score


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def MBE(y_true, y_pred):
    """
    Mean Bias Error, 平均偏差误差
    """
    return np.sum(y_true - y_pred) / y_true.size


def MAE(y_true, y_pred):
    """
    计算预测值的平均绝对误差(Mean Ablolute Error). 
    """
    return np.mean(np.abs(y_true - y_pred))


def MSE(y_true, y_pred):
    """
    计算预测值的均方误差(Mean Squared Error). 
    """
    return np.mean(np.square(y_pred - y_true))


def RMSE(y_true, y_pred):
    """
    计算预测值的均方根误差(Root Mean Square Error,). 
    """
    return np.sqrt(MSE(y_true, y_pred))


def RSE(y_true, y_pred):
    """
    计算预测值的相对平方误差(Relative Squared Error). 
    """
    return np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))


def NRMSE(y_true, y_pred):
    """
    normalized root mean squared error
    """
    squared_error = np.square((y_true - y_pred))
    sum_squared_error = np.sum(squared_error)
    rmse = np.sqrt(sum_squared_error / y_true.size)
    nrmse_loss = rmse / np.std(y_pred)

    return nrmse_loss


def RRMSE(y_true, y_pred):
    """
    relative root mean squarederror
    """
    num = np.sum(np.square(y_true - y_pred))
    den = np.sum(np.square(y_pred))
    squared_error = num / den
    rrmse_loss = np.sqrt(squared_error)

    return rrmse_loss


def RAE(y_true, y_pred):
    """
    Relative Absolute Error (RAE)
    """
    true_mean = np.mean(y_true)
    squared_error_num = np.sum(np.abs(y_true - y_pred))
    squared_error_den = np.sum(np.abs(y_true - true_mean))
    rae_loss = squared_error_num / squared_error_den

    return rae_loss


def MSLE(y_true, y_pred):
    """
    Mean Squared Logarithmic Error(MSLE)
    """
    square_error = np.square((np.log(y_true + 1) - np.log(y_pred + 1)))
    msle_error = np.mean(square_error)
    
    return msle_error


def RMSLE(y_true, y_pred):
    """
    Root Mean Squared Logarithmic Error (RMSLE)
    """
    square_error = np.square((np.log1p(y_true + 1) - np.log1p(y_pred + 1)))
    msle_error = np.mean(square_error)
    rmsle_loss = np.sqrt(msle_error)

    return rmsle_loss


# TODO
def MAPE(y_true, y_pred):
    """
    Mean Ablolute Percentage Error
    """
    if 1:
        if 0 in y_true:
            y_pred = pd.Series(y_pred)
            y_true = pd.Series(y_true)
            y_pred = y_pred[y_true != 0]
            y_true = y_true[y_true != 0]
        
        mape_loss = np.mean(np.abs((y_pred - y_true) / y_true))
    else:
        abs_error = np.abs(y_true - y_pred) / y_true
        sum_abs_error = np.sum(abs_error)
        mape_loss = (sum_abs_error / y_true.size) * 100

    return mape_loss


def WMAPE(y_true, y_pred):
    pass


def SMAPE(y_true, y_pred):
    pass


def Huber(y_true, y_pred, delta):
    """
    Huber Loss
    """
    huber_mse = 0.5 * np.square(y_true - y_pred)
    huber_mae = delta * (np.abs(y_true - y_pred) - 0.5 * (np.square(delta)))

    return np.where(np.abs(y_true - y_pred) <= delta, huber_mse, huber_mae)


# TODO
def LogCosh(y_true, y_pred, delta):
    diff = np.cosh(y_pred - delta)
    diff = np.log(diff)

    return diff.mean()


# TODO
def Quantile(y_true, y_pred):
    pass


# TODO
def DTW(y_true, y_pred):
    pass


# TODO
def TILDE_Q(y_true, y_pred):
    pass


def CORR(y_true, y_pred):
    """
    计算预测值的相关系数(Correlation). 
    """
    u = ((y_true - y_true.mean(0)) * (y_pred - y_pred.mean(0))).sum(0)
    d = np.sqrt(((y_true - y_true.mean(0)) ** 2 * (y_pred - y_pred.mean(0)) ** 2).sum(0))

    return (u / d).mean()


def MSPE(y_true, y_pred):
    """
    计算预测值的平均平方百分比误差(Mean Scaled Percentage Error). 
    """
    if 0 in y_true:
        y_pred = pd.Series(y_pred)
        y_true = pd.Series(y_true)
        y_pred = y_pred[y_true != 0]
        y_true = y_true[y_true != 0]

    return np.mean(np.square((y_pred - y_true) / y_true))


# TODO
def scoring():
    scoring_regressioner = {
        'R2': r2_score,
        'MES': mean_squared_error,
    }

    return scoring_regressioner



def main():
    pass

if __name__ == '__main__':
    main()
