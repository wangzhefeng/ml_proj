# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import AgglomerativeClustering

"""
class sklearn.cluster.AgglomerativeClustering(n_clusters = 2,
											  affinity = "euclidean",
											  memory = None,
											  connectivity = None,
											  compute_full_tree = "auto",
											  linkage = "ward",
											  pooling_func = "deprecated")
"""


# =======================================================================
# data
# =======================================================================
X = np.array([[1, 2],
			  [1, 4],
			  [1, 0],
			  [4, 2],
			  [4, 4],
			  [4, 0]])

X_test = np.array([[1, 3],
				   [2, 4]])

# =======================================================================
# clustering
# =======================================================================
clustering = AgglomerativeClustering()
clustering.fit(X)


print(clustering)
print(clustering.get_params())
print(clustering.labels_)
print(clustering.n_leaves_)
print(clustering.n_components_)
print(clustering.children_)

predictions = clustering.fit_predict(X_test)
print(predictions)
