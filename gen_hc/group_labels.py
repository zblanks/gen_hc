from math import ceil
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import networkx as nx
import community
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


def compute_kmc(V, k, random_state=None, n_jobs=1):
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
    >>> compute_kmc(V, 2)
    np.array([1, 0, 0])

    """
    rng = create_rng(random_state)
    kmeans = KMeans(n_clusters=k, random_state=rng, n_jobs=n_jobs)
    return kmeans.fit_predict(V)


def _check_valid_partition(partition, nlabels, metric):
    """
    Checks that a label grouping is valid (# of communities in {2, ..., labels})

    Parameters
    ----------
    partition : np.ndarray
        Proposed label grouping from community detection method
    nlabels : int
        Number of unique labels in the data
    metric: str
        Similarity measure used to generate partition

    Returns
    -------
    is_valid: bool
        Whether a partition is valid

    """
    ncommunities = len(np.unique(partition))
    if ncommunities < nlabels and ncommunities >= 2:
        return True
    else:
        metric_dict = {'rbf': 'RBF', 'l2': 'L2', 'linf': 'L-infinity'}
        raise ValueError('Could not find valid partition with {}. Try using '
                         'another kernel.'.format(metric_dict[metric]))


def _convert_format(partition):
    """
    Converts the format of the python-louvain into a numpy array

    Parameters
    ----------
    partition : dict
        Standard output from python-louvain package

    Returns
    -------
    partition: np.array
        Partition as a numpy array

    """
    return np.array([partition[val] for val in partition.keys()])


def compute_cd(V, metric='rbf', random_state=None, n_jobs=1):
    """
    Implements the community detection based approach for label grouping

    Parameters
    ----------
    V : np.ndarray
        Mean label point matrix
    metric : str
        Which measure to use to compute label similarity. Only support RBF
        ('rbf'), L2 ('l2'), and L-infinity ('linf') measures
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
    >>> V = np.array([[1, 1], [1.1, 1.1], [9.9, 9.9], [10, 10]])
    >>> compute_cd(V, 'rbf')
    np.array([0, 0, 1, 1])

    """
    # Only accept 'rbf', 'l2', or 'linf' as valid measures of label similarity
    if metric not in ['rbf', 'l2', 'linf']:
        raise ValueError('Invalid metric value; only accept RBF, L2, or L-inf '
                         'kernels')

    rng = create_rng(random_state)

    if metric == 'rbf':
        S = rbf_kernel(V)
    else:
        # The remaining kernels are distance measures and thus have to be
        # converted into a similarity metric
        dist_dict = {'l2': 'euclidean', 'linf': 'chebyshev'}
        D = pairwise_distances(V, metric=dist_dict[metric], n_jobs=n_jobs)
        S = 1 / np.exp(D)

    partition = community.best_partition(nx.Graph(S), random_state=rng)
    partition = _convert_format(partition)
    if _check_valid_partition(partition, V.shape[0], metric):
        return partition


def group_labels(X, y, group_algo='kmeans', random_state=None, k=None,
                 metric=None, n_jobs=1, label_frac=0.1):
    """
    Generates the label groupings

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    y : np.ndarray
        Label vector
    group_algo : str
        Which grouping algorithm to use -- 'kmeans' or 'cd'
    random_state : int or np.random.RandomState
        Pseudo random number generator
    k : int
        Number of meta-classes if k-means method is called
    metric : str
        Similarity metric to use if community detection method is called
    n_jobs : int
        Desired number of cores to run clustering algorithm; -1 runs all cores
    label_frac : float
        Fraction of the label space to use for meta-classes if the number
        is not specified

    Returns
    -------
    label_groups: np.ndarray
        Label grouping

    Examples
    --------
    >>> X = np.array([[1, 1],
                      [1.25, 1],
                      [1, 1.25],
                      [1.25, 1.25],
                      [5, 5],
                      [5.25, 5.25]])
    >>> y = np.array([0, 0, 1, 1, 2, 2])
    >>> group_labels(X, y, group_algo='kmeans', k=2)
    np.array([0, 0, 1])

    """
    X = StandardScaler().fit_transform(X)
    V = build_v_matrix(X, y)
    rng = create_rng(random_state)

    if group_algo == 'kmeans':
        if k is None:
            nlabels = V.shape[0]
            k = int(ceil(nlabels * label_frac))
        else:
            if not isinstance(k, int):
                raise TypeError('Number of meta-classes must be an integer')

        label_groups = compute_kmc(V, k, random_state=rng, n_jobs=n_jobs)
    elif group_algo == 'cd':
        if metric is None:
            metric = 'rbf'
        else:
            if not isinstance(metric, str):
                raise TypeError('Similarity metric must be a string with value'
                                'rbf, l2, or linf')
        label_groups = compute_cd(V, metric, random_state=rng, n_jobs=n_jobs)
    else:
        raise ValueError('Only accept "kmeans" and "cd" for grouping '
                         'algorithms')

    return label_groups
