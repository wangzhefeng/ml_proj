# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


# Settings
n_repeat = 50
n_train = 50
n_test = 1000
noise = 0.1
rng = np.random.RandomState(0)

# estimator
estimators = [
    ("Tree", DecisionTreeRegressor()),
    ("Bagging(Tree)", BaggingRegressor(base_estimator=DecisionTreeRegressor())),
]
n_estimators = len(estimators)

# Generate data
def f(x):
    x = x.ravel()
    return np.exp(-(x ** 2)) + 1.5 * np.exp(-(x - 2) ** 2)


def generate(n_samples, noise, n_repeat = 1):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X)
    if n_repeat == 1:
        y = f(X) + np.random.normal(0.0, noise, n_samples)
    else:
        y = np.zeros((n_samples, n_repeat))
        for i in range(n_repeat):
            y[:, i] = f(X) + np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))
    return X, y

# trian and test data
X_train = []
y_train = []
for i in range(n_repeat):
    X, y = generate(n_samples=n_train, noise = noise)
    X_train.append(X)
    y_train.append(y)

X_test, y_test = generate(n_samples=n_test, noise=noise, n_repeat=n_repeat)

print(X_train[0].shape)
print(len(X_train))
print(y_train[0].shape)
print(len(y_train))
print(X_test.shape)
print(y_test.shape)
print(X_test)
print(y_test)
# compare

