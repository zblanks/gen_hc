==============
Label Grouping
==============

The first step to employ the hierarchical classifiers (HCs) presented in this
code base is to group label "similar" labels together. Similarity can be
measured in a variety of ways, but in our research we have primarily focused
on :math:`L_2` distance, and the radial basis function (RBF) kernel, and
the :math:`L_\infty` distance. The :math:`L_2` norm is used for both grouping
methods whereas the RBF and :math:`L_\infty` norm is only applied for the
community detection approach.

Mean-Point Matrix
=================
There are two primary techniques we employ to group the labels -- a k-means
based approach and a community detection methodology. These algorithms were
selected because they can find solutions quickly and demonstrated good
empirical performance in a variety of experimental settings. For both methods
we have a two step process where we first find the mean representation of the
label, indicated as the :math:`\mathbf{V}` matrix and then apply the grouping
algorithm. We work with a mean point matrix versus the entire data set,
:math:`\mathbf{X}`, primarily for computational reasons. The mean point matrix
grows linearly with the number of labels whereas :math:`\mathbf{X}` grows both
with the number of samples as well as the number of classes. To find
:math:`\mathbf{V}`, suppose we have data
:math:`\mathcal{D} = (\mathbf{X}, \mathbf{y})` which equals

.. math::
    \mathbf{X} =
        \begin{bmatrix}
            1    & 1 \\
            1.25 & 1 \\
            1    & 1.25 \\
            1.25 & 1.25 \\
            5    & 5   \\
            5.25 & 5.25 \\
        \end{bmatrix}

and

.. math::
    \mathbf{y} = (0, 0, 1, 1, 2, 2)^T.

We can then employ the function :func:`~gen_hc.group_labels.build_v_matrix`
like

.. ipython::

    In [1]: import numpy as np

    In [2]: from gen_hc.group_labels import build_v_matrix

    In [3]: X = np.array([[1, 1], [1.25, 1], [1, 1.25], [1.25, 1.25], [5, 5],
       ...:               [5.25, 5.25]])

    In [4]: y = np.array([0, 0, 1, 1, 2, 2])

    @doctest
    In [5]: build_v_matrix(X, y)
    Out[5]:
    array([[1.125, 1.   ],
           [1.125, 1.25 ],
           [5.125, 5.125]])

and the output will yield the :math:`\mathbf{V}` matrix. This matrix is the
base input for both of the label grouping approaches.

K-Means Clustering Approach
===========================
The simplest of the two label grouping methods is to employ k-means clustering
on :math:`\mathbf{V}` by specifying the desired number of meta-classes. For
example,

.. ipython:: python

    from gen_hc.group_labels import compute_kmc

    V = np.array([[1, 1], [2.5, 2.5], [3, 3]])
    compute_kmc(V, k=2)

goes through the specified :math:`\mathbf{V}` matrix containing three labels
and finds a partition which groups them into two meta-classes. One can also
access this method more generally through the
:func:`~gen_hc.group_labels.group_labels`. For example,

.. ipython::

    In [6]: from gen_hc import group_labels

    In [7]: X = np.array([[0.9, 0.9], [1.1, 1.1], [2.4, 2.4], [2.6, 2.6],
       ...:               [2.9, 2.9], [3.1, 3.1]])

    In [8]: y = np.array([0, 0, 1, 1, 2, 2])

    @doctest
    In [9]: group_labels(X, y, group_algo='kmeans', k=2)
    Out[9]: array([1, 0, 0], dtype=int32)

where :func:`~gen_hc.group_labels.group_labels` is calling the k-means
clustering approach and finding a partition with two meta-classes.

Community Detection Approach
============================
The other approach that is supported when grouping labels is a community
detection approach. For this methodology, :math:`\mathbf{V}` is converted into
a similarity matrix, :math:`\mathbf{S}`, which can be viewed as a graph, and
then the `Louvain method <https://arxiv.org/abs/0803.0476>`_ is applied to
partition the nodes into communities.

The key hyper-parameter for this approach is the similarity metric to employ.
We support the RBF kernel, :math:`L_2` distance, and :math:`L_\infty` distance,
however, for the distance measures, we employ a trick

.. math::

    s_{ij} = \frac{1}{\exp(d_{ij})}

where :math:`d_{ij}` is the distance between labels :math:`i` and :math:`j`.
The kernel that typically performed the best was the RBF measure so it is the
default value. The grouping method can be accessed by typing, for example,

.. ipython::

    In [10]: from gen_hc.group_labels import compute_cd

    In [11]: V = np.array([[1, 1], [1.1, 1.1], [9.9, 9.9], [10, 10]])

    @doctest
    In [12]: compute_cd(V, metric='rbf')
    Out[12]: array([0, 0, 1, 1])

Alternatively, one can also again access the more general function,
:func:`~gen_hc.group_labels.group_labels` and have

.. ipython::

    In [13]: X = np.array([[.9, .9], [1., 1.], [1.05, 1.05], [1.15, 1.15],
       ....:               [9.8, 9.8], [10., 10.], [9.9, 9.9], [10.1, 10.1]])

    In [14]: y = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    @doctest
    In [15]: group_labels(X, y, group_algo='cd', metric='rbf')
    Out[15]: array([0, 0, 1, 1])

.. automodule:: gen_hc.group_labels
    :members:
