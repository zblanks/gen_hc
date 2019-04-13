import numpy as np
import pytest
from .. import group_labels


def test_build_v_matrix():
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y = np.array([0, 0, 1, 1])
    V = group_labels.build_v_matrix(X, y)
    assert np.allclose(V, np.array([[1.5, 1.5], [3.5, 3.5]]))


def test_compute_kmc():
    V = np.array([[1, 1], [2.5, 2.5], [3, 3]])
    label_groups = group_labels.compute_kmc(V, 2)
    assert np.allclose(label_groups, np.array([1, 0, 0]))


def test_check_valid_partition():
    # Check error is raised with invalid partition
    partition = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        group_labels._check_valid_partition(partition, 3, 'rbf')

    # Check that a valid partition is good
    partition = np.array([0, 1, 1])
    assert group_labels._check_valid_partition(partition, 3, 'rbf') == True


def test_convert_format():
    # Check that the function can correctly convert the output
    partition = {0: 0, 1: 0, 2: 1}
    correct_arr = np.array([0, 0, 1])
    assert np.allclose(group_labels._convert_format(partition), correct_arr)


def test_compute_cd():
    # Check that RBF works
    V = np.array([[1, 1], [1.1, 1.1], [9.9, 9.9], [10, 10]])
    partition = group_labels.compute_cd(V, 'rbf')
    assert np.allclose(partition, np.array([0, 0, 1, 1]))

    # Check L2 works
    partition = group_labels.compute_cd(V, 'l2')
    assert np.allclose(partition, np.array([0, 0, 1, 1]))

    # Check L-infinity works
    partition = group_labels.compute_cd(V, 'linf')
    assert np.allclose(partition, np.array([0, 0, 1, 1]))

    # Check error is thrown with invalid community with each as own community
    V = np.array([[0, 0], [5, 5], [10, 10]])
    with pytest.raises(ValueError):
        group_labels.compute_cd(V, 'rbf')

def test_group_labels():
    # Check that the example works
    X = np.array([[1, 1],
                  [1.25, 1],
                  [1, 1.25],
                  [1.25, 1.25],
                  [5, 5],
                  [5.25, 5.25]])
    y = np.array([0, 0, 1, 1, 2, 2])
    label_groups = group_labels.group_labels(X, y, group_algo='kmeans', k=2)
    assert np.allclose(label_groups, np.array([0, 0, 1]))

    # Check that error is thrown when wrong grouping algorithm is passed
    with pytest.raises(ValueError):
        group_labels.group_labels(X, y, group_algo='meow')
