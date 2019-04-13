import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances
import networkx as nx
from .utils import create_rng


def build_v_matrix(X, y):
    """
    Computes mean label point matrix

    The mean label point matrix is used for all of the label grouping methods
    because it is able to represent the label space in a computationally
    tractable manner vice doing pairwise computations for all of the samples
    that belong to each label.

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    y : np.ndarray
        Label vector

    Returns
    -------
    V : np.ndarray
        Mean label point matrix

    Examples
    --------
    >>> X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> build_v_matrix(X, y)
    np.array([[1.5, 1.5],
              [3.5, 3.5]]])

    """

    uniq_labels = np.unique(y)
    nclasses = len(uniq_labels)
    p = X.shape[1]
    V = np.empty(shape=(nclasses, p))

    idx_list = [np.where(y == i)[0] for i in uniq_labels]
    for i in range(nclasses):
        V[i, :] = X[idx_list[i], :].mean(axis=0)

    return V


def compute_kmeans_group(V, k, random_state=None, n_jobs=1):
    """
    Implements the K-means based approach to grouping labels

    Parameters
    ----------
    V : np.ndarray
        Mean label point matrix
    k : int
        Desired number of meta-classes
    random_state : int or np.random.RandomState
        Pseudo random number generator
    n_jobs : int
        Desired number of cores to run clustering algorithm; -1 runs all cores

    Returns
    -------
    label_groups: np.ndarray
        Label grouping

    Examples
    --------
    >>> V = np.array([[1, 1], [2.5, 2.5], [3, 3]])
    >>> compute_kmeans_group(V, 2)
    np.array([1, 0, 0])

    """
    rng = create_rng(random_state)
    kmeans = KMeans(n_clusters=k, random_state=rng, n_jobs=n_jobs)
    return kmeans.fit_predict(V)
