from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_erro




elastic_net = ElasticNet(alpha = 0.1, l1_ratio = 0.5) # r=l1_ratio\n
elastic_net.fit(X_train, y_train)
y_pred = elastic_net.predict(X_test)
scores = mean_squared_error(y_test, y_pred)
