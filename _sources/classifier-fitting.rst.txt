=================================
Classifier Fitting and Prediction
=================================

Once the label grouping has been learned (or if it is known beforehand), the
next step is to fit the model using the data
:math:`\mathcal{D} = (\mathbf{X}, \mathbf{y})` and tree, as defined by the
matrix, :math:`\mathbf{Z}`. The simplest way to approach this problem is to
train each of the parent nodes in the tree separately from one another. This
allows the algorithm to be parallelized and avoids the issue of having to
globally optimize the HC. Additionally, once the tree has been fit to the
provided data, one needs to be able to compute test set predictions. In this
package we utilize a way that avoids the "routing problem" -- the challenge
where issues made at higher levels of the tree are propagated through the
system.

Fitting the Classifier
======================
The primary interface in which one can both find a label grouping for the data
and fit an HC, is via the :class:`GenHC` class. This class
allows the user to control how the label grouping is performed (or if desired
the user can provide their own partition assuming it is valid), define how
the test set predictions are calculated, and how the classifier should fit
the data. Let's start with a simple example:

.. ipython::
    :okwarning:

    In [1]: from sklearn.datasets import make_classification

    In [2]: from gen_hc import GenHC

    In [3]: X, y = make_classification(n_samples=10, n_classes=3, n_features=10,
       ...:                            n_informative=10, n_redundant=0,
       ...:                            n_repeated=0, n_clusters_per_class=1,
       ...:                            random_state=17)

    In [4]: hc = GenHC()

    In [5]: hc.fit(X, y, clf='rf', n_estimators=10)

    @doctest
    In [6]: hc.predict(X)
    Out[6]: array([2, 0, 0, 1, 0, 0, 0, 2, 1, 2])

In this example, we created a synthetic data set, defined an instance of the
:class:`GenHC` class, fit the model to the provided data, and
finally generated predictions using the law of total probability (LOTP) method.
Let's walk through the last parts of the code block to explain what is
happening underneath the hood.

When the :func:`~gen_hc.gen_hc.GenHC` class is called, it is not necessary to
provide any arguments. The class has been provided sensible default values
which correspond to what has been observed to work in practical settings.
Therefore if one were to simply use :func:`~gen_hc.gen_hc.GenHC` as is,
the algorithm would default to use the k-means approach (more information
provided in :ref:`K-Means Clustering Approach`). Moreover, the algorithm will
default to 10% of the number of labels or two meta-classes -- whichever is
larger. Finally, it will also default to the LOTP generated predictions
(discussed more in :ref:`Calculating Predictions`) and will have the number of
features be the number of labels passed to the method minus one. The features
are adjusted in the HC because it has been demonstrated that giving different
sets of features to the parent nodes in the graph improves downstream
performance.

When fitting the an HC using the :func:`~gen_hc.gen_hc.GenHC.fit` method, a
thing to be careful about is that the user must provide the classifier as well
as any keyword arguments affiliated with the estimator. At the moment, we only
support Scikit-learn classifiers that can output a posterior probability when
calculating test predictions. In principle though, this framework is quite
general and as long as the classifier given to the fit method can produce
probabilistic predictions, then it is a valid model for this type of
classifier. For example, if one wanted to use a convolutional neural network
(CNN), with appropriate modifications to the code (namely adjusting the methods
to calculate predictions and how the code base interacts with the classifier),
this would work for a generalized HC. Moreover at the moment, the classifier
provided with the :func:`~gen_hc.gen_hc.GenHC.fit` method is the one that is
used for every parent node in the tree. This is done for simplicity, but
strictly speaking there is no reason that this is a hard requirement.
Again the code base could be adjusted to allow the user greater flexibility
in specifying the classifier for each node in the graph. One potential use
case is that perhaps certain subsets of the labels are easier to predict and
thus a less powerful model could be used to reduce the computational burden
of fitting the HC.

Calculating Predictions
=======================
After the model has been fit, there are two ways in which one can calculate
test set predictions: the standard "arg-max" approach and one which uses
the LOTP. It is strongly recommended to use the LOTP method because it has
been shown to give more stable posterior predictions and due to its NumPy based
implementation is not that much slower than the traditional "arg-max" method.

Arg-Max Predictions
-------------------
The standard way to predict classes with an HC is to start at the top of the
tree and proceed in the direction which gives the highest posterior probability
for that parent node. Certain papers have shown that this technique can reduce
the time it takes to generate test set predictions such as
:cite:`bengio2010label`. If this is the desired way to generate predictions
it can be done by typing

.. ipython::
    :okwarning:

    In [7]: hc = GenHC(prediction_method='argmax')

    In [8]: hc.fit(X, y, clf='rf', n_estimators=10)

    @doctest
    In [9]: hc.predict(X)
    Out[9]: array([2, 0, 0, 1, 0, 0, 0, 2, 1, 2])

Note that this approach will not allow the user to create a probability matrix
:math:`\hat{\mathbf{Y}}` because there is no clear way in which one could
generate this prediction without claiming that a majority of the labels have
zero posterior probability mass.

LOTP Predictions
----------------
The alternative way to generate predictions, and the one that is strongly
recommended, is to calculate test set values using the LOTP. Formally this
equates to solving

.. math::

\mathbb{P}\left(y_i = j \mid \mathbf{x}_i \right) = \sum_{l=1}^L
\mathbb{P}\left(y_i \in \mathbf{Z}_l \mid \mathbf{x}_i\right)
\mathbb{P}\left(y_i = j \mid \mathbf{x}_i, y_i \in \mathbf{Z}_l\right) \quad
\forall j

where :math:`j` denotes the label whose probability is being predicted from the
set of all classes and :math:`L` is the total number of meta-classes which
has been learned. This equations assumes that the tree has only one layer, but
one could write a recursive form of this equation if the user desired to relax
this constraint.

This is default method for generating predictions for the HC, but it can
be explicitly called by typing

.. ipython::
    :okwarning:

    In [10]: hc = GenHC(prediction_method='lotp')

    In [11]: hc.fit(X, y, clf='rf', n_estimators=10)

    @doctest
    In [12]: hc.predict(X)
    Out[12]: array([2, 0, 0, 1, 0, 0, 0, 2, 1, 2])

Additionally, the benefit of using this approach is that it also supports
creating the :math:`\hat{\mathbf{Y}}` matrix which can be accessed through
the :func:`~gen_hc.gen_hc.GenHC.predict_proba` method.

.. ipython::

    @doctest
    In [13]: hc.predict_proba(X)
    Out[13]:
    array([[0.09, 0.01, 0.9 ],
           [1.  , 0.  , 0.  ],
           [0.9 , 0.1 , 0.  ],
           [0.  , 1.  , 0.  ],
           [1.  , 0.  , 0.  ],
           [1.  , 0.  , 0.  ],
           [0.5 , 0.5 , 0.  ],
           [0.1 , 0.  , 0.9 ],
           [0.  , 1.  , 0.  ],
           [0.09, 0.01, 0.9 ]])

.. autoclass:: gen_hc.GenHC
    :members:

.. bibliography:: references.bib
