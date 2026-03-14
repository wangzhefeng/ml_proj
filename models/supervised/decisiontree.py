# -*- coding: utf-8 -*-

# ***************************************************
# * File        : decisiontree_demo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-05
# * Version     : 0.1.040522
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


dt_gini = DecisionTreeClassifier(
    criterion = "gini",
    splitter = "best",
    max_depth = None,
    min_samples_split = 2,
    min_samples_leaf = 1,
    min_weight_fraction_leaf = 0.0,
    max_features = None,
    random_state = None,
    max_leaf_nodes = None,
    min_impurity_decrease = 0.0,
    min_impurity_split = None,
    class_weight = None,
    # presort = False
)

dt_entropy = DecisionTreeClassifier(
    criterion = "entropy",
    splitter = "best",
    max_depth = None,
    min_samples_split = 2,
    min_samples_leaf = 1,
    min_weight_fraction_leaf = 0.0,
    max_features = None,
    random_state = None,
    max_leaf_nodes = None,
    min_impurity_decrease = 0.0,
    min_impurity_split = None,
    class_weight = None,
    # presort = False
)

# ------------------------------
# data
# ------------------------------
iris = datasets.load_iris()

# ------------------------------
# hold out split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, 
    iris.target,
    test_size = 0.3,
    random_state = 27149
)
dt_gini.fit(X_train, y_train)

# ------------------------------
# out put
# ------------------------------
# attributes
print(dt_gini.classes_)
print(dt_gini.n_classes_)
print(dt_gini.n_features_)
print(dt_gini.n_outputs_)
print(dt_gini.feature_importances_)
print(dt_gini.max_features_)
print(dt_gini.tree_)


# methods
preds = dt_gini.predict(X_test)
preds_proba = dt_gini.predict_proba(X_test)
preds_log_proba = dt_gini.predict_log_proba(X_test)

print(preds)
print(preds_proba)
print(preds_log_proba)

accuracy = dt_gini.score(X_test, y_test)
print(accuracy)

print(dt_gini.decision_path(X_train))
print(dt_gini.decision_path(X_test))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
