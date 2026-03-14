# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cluster
from sklearn import mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)


# ======================================================================================
# data
# ======================================================================================
n_samples = 1500
noise_circels = datasets.make_circles(n_samples = n_samples, factor = 0.5, noise = 0.05)
noise_moons = datasets.make_moons(n_samples = n_samples, noise = 0.05)
blobs = datasets.make_blobs(n_samples = n_samples, random_state = 8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples = n_samples, random_state = random_state)
transformation = [[0.6, -0.6],
                  [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples = n_samples, cluster_std = [1.0, 2.5, 0.5], random_state = random_state)



# ---------------------------------------------------------------
# cluser parameters
# ---------------------------------------------------------------
default_base = {
    'quantile': 0.3,
    'eps': 0.3,
    'damping': 0.9,
    'preference': -200,
    'n_neighbors': 10,
    'n_clusters': 3
}

dataset = [
    (noise_circels, {'damping': 0.77, 'preference': -240, 'quantile': 0.2, 'n_clusters': 2}),
    (noise_moons, {'damping': 0.75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': 0.18, 'n_neighbors': 2}),
    (aniso, {'eps': 0.15, 'n_neighbors': 2}),
    (blobs, {}),
    (no_structure, {})
]






# ---------------------------------------------------------------
#
# ---------------------------------------------------------------
plt.figure(figsize=(9 * 1.3 + 2, 14.5))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.96,
					wspace=0.05, hspace=0.01)

for i_dataset, (dataset, algo_params) in enumerate(dataset):
	params = default_base.copy()
	params.update(algo_params)

	X, y = dataset
	X = StandardScaler().fit_transform(X)

	# Algorithms
	bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
	ms = cluster.MeanShift(bandwidth = bandwidth, bin_seeding = True)
	two_means = cluster.MiniBatchKMeans(n_clusters = params['n_clusters'])
	# connectivity matrix for structured Ward
	connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)
	connectivity = 0.5 * (connectivity + connectivity.T)

	ward = cluster.AgglomerativeClustering(n_clusters = params['n_clusters'],
										   linkage='ward',
										   connectivity = connectivity)
	average_linkage = cluster.AgglomerativeClustering(n_clusters = params['n_clusters'],
													  linkage = 'average',
													  affinity = 'cityblock')
	complete_linkage = cluster.AgglomerativeClustering(n_clusters = params['n_clusters'],
													   linkage = 'complete')
	single_linkage = cluster.AgglomerativeClustering(n_clusters = params['n_clusters'],
													 linkage='single')
	spectral = cluster.SpectralClustering(n_clusters = params['n_cluster'],
										  eigen_solver='arpack',
										  affinity = 'nearest_neighbors')
	dbscan = cluster.DBSCAN(eps = params['eps'])
	affinity_propagation = cluster.AffinityPropagation(damping = params['damping'], preference = params['preference'])
	birch = cluster.Birch(n_clusters = params['n_clusters'])
	gmm = mixture.GaussianMixture(n_components = params['n_clusters'], covariance_type='full')

	clustering_algorithms = {
		('MeanShift', ms),
		('MiniBatchKMeans', two_means),
		('Ward', ward),
		('AgglomerativeClustering', average_linkage),
		('AgglomerativeClustering', complete_linkage),
		('AgglomerativeClustering', single_linkage),
		('SpectralClustering', spectral),
		('DBSCAN', dbscan),
		('AffinityPropagation', affinity_propagation),
		('Birch', birch),
		('GaussianMixture', gmm)
	}

	for name, algorithm in clustering_algorithms:
		algorithm.git(X)

		if hasattr(algorithm, 'labels_'):
			y_pred = algorithm.labels_.astype(np.int)
		else:
			y_pred = algorithm.predict(X)



from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans


class kmeans_clustering:
	def __init__(self, X, n_clusters, init_centers, n_init,
				 max_iter, batch_size, tol, precompute_distances, verbose,
				 random_state, copy_x, n_jobs, algorithm):
		self.X = X
		self.n_clusters = n_clusters
		self.init_centers = init_centers
		self.n_init = n_init
		self.max_iter = max_iter
		self.tol = tol
		self.precompute_distances = precompute_distances
		self.verbose = verbose
		self.random_state = random_state
		self.copy_x = copy_x
		self.n_jobs = n_jobs
		self.algorithm = algorithm
		self.batch_size = batch_size

	def kmeans(self):
		kmeans_params = {
			"n_clusters": self.n_clusters,
			"init": self.init_centers,
			"n_init": self.n_init,
			"max_iter": self.max_iter,
			"tol": self.tol,
			"precompute_distances": self.precompute_distances,
			"verbose": self.verbose,
			"random_state": self.random_state,
			"copy_x": self.copy_x,
			"n_jobs": self.n_jobs,
			"algorithm": self.algorithm
		}
		cluster = KMeans(**kmeans_params)
		cluster.fit(self.X)

		return cluster


	def mini_batch_kmeans(self):
		mini_batch_kmeans_params = {
			"n_clusters": self.n_clusters,
			"init": self.init_centers,
			"max_iter": self.max_iter,
			"batch_size": self.batch_size,
			"tol": self.tol,
			"precompute_distances": self.precompute_distances,
			"verbose": self.verbose,
			"random_state": self.random_state,
			"copy_x": self.copy_x,
			"n_jobs": self.n_jobs,
			"algorithm": self.algorithm
		}
		cluster = MiniBatchKMeans(**mini_batch_kmeans_params)

		cluster.fit(self.X)

		return cluster
