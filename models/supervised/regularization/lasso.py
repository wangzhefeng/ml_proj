from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import mean_squared_error



lasso_reg = Lasso(alpha = 0.1)
lasso_reg.fit(X_train, y_train)
y_pred = lasso_reg.predict(X_test)
scores = mean_squared_error(y_test, y_pred)


lasso_sgd_reg = SGDRegressor(penalty = "l1")
lasso_sgd_reg.fit(X_train, y_train)
y_pred = lasso_sgd_reg.predict(X_test)
scores = mean_squared_error(y_test, y_pred)