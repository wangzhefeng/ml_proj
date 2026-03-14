

from sklearn.base import clone
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

def earlyStopping(model, X_train, y_train, X_val, y_val, n_iters):
    # training and 正则化
    minimum_val_error = float("intf")
    best_epoch = None
    best_model = None

    for epoch in range(n_iters): # 0 ~ n_iters-1
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        val_error = mean_squared_error(y_val_pred, y_val)
        if val_error < minimum_val_error:
            minimum_val_error = val_error
            best_epoch = epoch
            best_model = clone(sgd_reg)

    return best_epoch, best_model

