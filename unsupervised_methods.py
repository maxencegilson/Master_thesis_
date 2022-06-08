"""
Master Thesis
Academic year 2021-2022

Authors:
    - GILSON Maxence

Credit :
    - The different clustering where already created and available on different repositories imported
"""

###########
# Imports #
###########

# Clustering techniques
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster import BisectingKMeans
from sklearn_som.som import SOM


def get_name_method(method):
    if method == 1:
        return "K-Modes"
    if method == 2:
        return "K-Means"
    if method == 3:
        return "Bisecting K-Means"
    if method == 4:
        return "Self-organizing maps"


def kmodes_method(DB, n_clusters, max_iter, n_init, predict):
    """
	Returns the clusters created by the k-modes algorithm

	Parameters
    -----------
    n_clusters :int, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter :300
        Maximum number of iterations of the k-modes algorithm for a
        single run.v
    n_init : int, default: 10
        Number of time the k-modes algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of cost.
    predict : bool
		If true : fit and predict the database otherwise only fit it
	"""

    clusters = KModes(n_clusters=n_clusters, max_iter=max_iter, init='Huang', n_init=n_init)

    """
	Parameters
    n_clusters : defined above
    max_iter : defined above
    init : {'Huang', 'Cao', 'random' or an ndarray}, default: 'Cao'
        Method for initialization:
        'Huang': Method in Huang [1997, 1998]
        'Cao': Method in Cao et al. [2009]
        'random': choose 'n_clusters' observations (rows) at random from
        data for the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centroids.
    n_init : defined above

	"""
    if predict:
        clusters = clusters.fit_predict(DB)
    else:
        clusters = clusters.fit(DB)
    return clusters


def kmeans_method(DB, n_clusters, max_iter, n_init, predict):
    """
	Returns the clusters created by the k-means algorithm

	Parameters
    Same as k-modes
	"""

    clusters = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)

    """
	Parameters
    n_clusters : as defined above
	  init : {‘k-means++’, ‘random’}, 
	  	  'k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
	  	  ‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.
    n_init : as defined above
    max_iter : as defined above
	"""

    if predict:
        clusters = clusters.fit_predict(DB)
    else:
        clusters = clusters.fit(DB)
    return clusters


def bisecting_kmeans_method(DB, n_clusters, max_iter, n_init, predict):
    """
    Returns the clusters created by the bisecting k-means algorithm

    Parameters
    Same as k-modes
    """

    clusters = BisectingKMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)

    if predict:
        clusters = clusters.fit_predict(DB)
    else:
        clusters = clusters.fit(DB)
    return clusters


def som_method(DB, n_clusters, max_iter, predict):
    """
	Returns the clusters created by the self-organising maps algorithm

	Parameters
    Same as k-modes
	"""
    DB_np = DB.to_numpy()
    dim = DB_np.shape[1]
    if n_clusters == 4:
        m = 2
        n = 2
    elif n_clusters == 6:
        m = 3
        n = 2
    elif n_clusters == 8:
        m = 4
        n = 2
    elif n_clusters == 9:
        m = 3
        n = 3
    elif n_clusters == 10:
        m = 5
        n = 4
    else:
        m = n_clusters
        n = 1
    clusters = SOM(m=m, n=n, dim=dim, lr=1, sigma=1, max_iter=max_iter)
    """
    Parameters
    m, n : int 
        The dimension of the created map
    dim : int  
        The number of features in the input dataset
    """
    if predict:
        clusters = clusters.fit_predict(DB_np, epochs=50)
    else:
        clusters.fit(DB_np, epochs=50)
    return clusters



def methods(DB, method, n_clusters, max_iter, n_init, predict):
    """
    Returns the clusters created by the called algorithm
	Parameters
	----------
	method : int
		The number of the method used
			- 1 = k-modes
			- 2 = k-means
			- 3 = bisecting k-means
			- 4 = Self-organizing maps

    n_clusters: as defined above
    max_iter: as defined above
    n_init: as defined above
	predict : as defined above
	"""

    if method == 1:
        clusters = kmodes_method(DB, n_clusters, max_iter, n_init, predict)

    if method == 2:
        clusters = kmeans_method(DB, n_clusters, max_iter, n_init, predict)

    if method == 3:
        clusters = bisecting_kmeans_method(DB, n_clusters, max_iter, n_init, predict)

    if method == 4:
        clusters = som_method(DB, n_clusters, max_iter, predict)

    return clusters
