import numpy as np
from ..group_labels import build_v_matrix, compute_kmeans_group


def test_build_v_matrix():
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y = np.array([0, 0, 1, 1])
    V = build_v_matrix(X, y)
    assert np.allclose(V, np.array([[1.5, 1.5], [3.5, 3.5]]))


def test_compute_kmeans_group():
    V = np.array([[1, 1], [2.5, 2.5], [3, 3]])
    label_groups = compute_kmeans_group(V, 2)
    assert np.allclose(label_groups, np.array([1, 0, 0]))
