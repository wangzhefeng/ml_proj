from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error


ridge_reg = Ridge(alpha = 1,solver = "cholessk") # 利用Andre-Louis Cholesky矩阵分解法
ridge_reg.fit(X_train, y_train)
y_pred = ridge_reg.predict(X_test)
scores = mean_squared_error(y_test, y_pred)


ridge_sgd_reg = SGDRegressor(penalty = "l2")
ridge_sgd_reg.fit(X_train, y_train)
y_pred = ridge_sgd_reg.predict(X_test)
scores = mean_squared_error(y_test, y_pred)