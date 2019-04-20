from math import floor
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.base import is_classifier
from .group_labels import group_labels
from joblib import Parallel, delayed
from .utils import create_rng


class GenHC:
    """
    Interface to fit and get predictions for a generalized hierarchical classifier

    Parameters
    ----------
    group_algo : {'kmeans', 'cd'}, optional
        Which grouping algorithm to use
    k : int, optional
        Number of meta-classes if k-means method is called
    label_frac : float, optional
        Fraction of the label space to use for meta-classes if the number
        is not specified
    metric : {'rbf', 'l2', 'linf}, optional
        Similarity metric to use if community detection method is called
    label_groups : np.ndarray[num_classes,], optional
        Pre-determined label groupings where each label is assigned to only
        one meta-class
    prediction_method : {'lotp', 'argmax'}, optional
        Method by which predictions are generated for the HC
    num_features : int, optional
        Number of features to use to train the model
    feature_frac : float, optional
        Fraction of features to use with PCA if LDA is not feasible for the data
    n_jobs : int, optional
        Desired number of cores to run algorithm; -1 runs all cores
    random_state : int or np.random.RandomState, optional
        Pseudo random number generator

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from gen_hc import GenHC
    >>> X, y = make_classification(n_samples=10, n_classes=3, n_features=10,
    ...                            n_informative=10, n_redundant=0, n_repeated=0,
    ...                            n_clusters_per_class=1, random_state=17)
    >>> hc = GenHC()
    >>> hc.fit(X, y, clf='rf', n_estimators=10)
    >>> hc.predict(X)
    array([2, 0, 0, 1, 0, 0, 0, 2, 1, 2])
    """

    def __init__(self, group_algo='kmeans', k=None, label_frac=0.10,
                 metric=None, label_groups=None, prediction_method='lotp',
                 num_features=None, feature_frac=0.25, n_jobs=1,
                 random_state=None):

        self.group_algo = group_algo
        self.k = k
        self.label_frac = label_frac
        self.metric = metric
        self.label_groups = label_groups
        self.prediction_method = prediction_method
        self.num_features = num_features
        self.feature_frac = feature_frac
        self.n_jobs = n_jobs
        self.random_state = create_rng(random_state)

        self.models = None

        if self.label_groups is not None:
            self.num_metaclasses = len(np.unique(self.label_groups))
            self._check_valid_grouping()
            self.non_lone_leaves = self._get_leaf_idx()
        else:
            self.num_metaclasses = None

    def _check_is_trained(self):
        """
        Checks whether the HC has been trained

        Returns
        -------
        None

        """
        if self.models is None:
            raise ValueError('Need to train HC before generating predictions')
        else:
            return None

    def node_predict_proba(self, X):
        """
        Gets the node-level probability predictions

        Parameters
        ----------
        X : np.ndarray[num_samples, num_features]
            Test data

        Returns
        -------
        Y : np.ndarray[num_samples, num_meta-classes]
            Node-level probability prediction matrix

        """
        # It is assumed from training that the top classifier is in the -1
        # index
        self._check_is_trained()
        return self.models[-1].predict_proba(X)

    def _get_leaf_idx(self):
        """
        Gets the indices of leaves that are not by themselves

        Returns
        -------
        leaf_idx : np.ndarray
            Non-lone labels at the leaves of the tree

        """
        return np.where(np.bincount(self.label_groups) > 1)[0]

    def _leaf_predict_proba(self, X, metaclass_idx):
        """
        Gets the leaf-level predictions for a given meta-class

        Parameters
        ----------
        X : np.ndarray[num_samples, num_features]
            Test data
        metaclass_idx : int
            Meta-class index in self.models

        Returns
        -------
        Y : np.ndarray[num_samples, num_classes]
            Node-level probability prediction matrix

        """
        Y = self.models[metaclass_idx].predict_proba(X)
        return Y

    # noinspection PyCompatibility
    def _lotp_predict_proba(self, X):
        """
        Get the Law of Total Probability predictions for the HC

        Parameters
        ----------
        X : np.ndarray[num_samples, num_features]
            Test data

        Returns
        -------
        Y : np.ndarray[num_samples, num_classes]
            Leaf-level probability prediction matrix

        """

        num_samples = X.shape[0]
        num_classes = len(self.label_groups)
        lone_leaves = np.setdiff1d(np.arange(self.num_metaclasses),
                                   self.non_lone_leaves)

        N = self.node_predict_proba(X)

        L = [np.array([])] * self.num_metaclasses
        for val in lone_leaves:
            L[val] = np.ones(shape=(num_samples, 1))

        for (i, val) in enumerate(self.non_lone_leaves):
            L[val] = self._leaf_predict_proba(X, i)

        Y = np.empty(shape=(num_samples, num_classes))
        for i in range(self.num_metaclasses):
            idx = np.where(self.label_groups == i)[0]
            Y[:, [*idx]] = N[:, [i]] * L[i]

        return Y

    def _adjust_labels(self, y, metaclass_idx):
        """
        Adjusts the labels to get back to their original values

        Parameters
        ----------
        y : np.ndarray[num_samples,]
            Label vector
        metaclass_idx : int
            Index which identifies the metaclass used for adjusting the labels

        Returns
        -------
        y_new : np.ndarray[num_samples,]
            Adjusted label vector

        """
        relevant_labels = np.where(self.label_groups == metaclass_idx)[0]
        current_labels = np.unique(y)
        label_map = dict(zip(current_labels, relevant_labels))
        return [label_map[val] for val in y]

    def _argmax_predict(self, X):
        """
        Gets the argmax predictions

        Parameters
        ----------
        X : np.ndarray[num_samples, num_features]
            Test data

        Returns
        -------
        y : np.ndarray[num_samples,]
            Vector of class labels

        """

        Y = self.node_predict_proba(X)
        y_node = Y.argmax(axis=1)
        y = np.empty_like(y_node)

        lone_leaves = np.setdiff1d(np.arange(self.num_metaclasses),
                                   self.non_lone_leaves)

        for val in lone_leaves:
            idx = np.where(y_node == val)[0]
            y_tmp = y_node[idx]

            # y_tmp corresponds to the meta-class label and so we have to map
            # it back to its original label value
            y_tmp = self._adjust_labels(y_tmp, val)
            y[idx] = y_tmp

        for (i, val) in enumerate(self.non_lone_leaves):
            idx = np.where(y_node == val)[0]
            y_tmp = self._leaf_predict_proba(X[idx], i).argmax(axis=1)

            # The labels for the meta-class are re-mapped to 0 to N and so have
            # to be adjusted back to their original values
            y_tmp = self._adjust_labels(y_tmp, val)
            y[idx] = y_tmp

        return y

    def predict(self, X):
        """
        Predict the class labels for the provided data

        Parameters
        ----------
        X : np.ndarray[num_samples, num_features]
            Test data

        Returns
        -------
        y : np.ndarray[num_samples,]
            Vector of class labels

        """
        self._check_is_trained()

        if self.prediction_method is 'lotp':
            Y = self._lotp_predict_proba(X)
            return Y.argmax(axis=1)
        else:
            return self._argmax_predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for X

        Parameters
        ----------
        X : np.ndarray[num_samples, num_features]
            Test data

        Returns
        -------
        Y : np.ndarray[num_samples, num_classes]
            Leaf-level probability prediction matrix

        """
        self._check_is_trained()

        if self.prediction_method is 'argmax':
            raise ValueError('Cannot generate probabilistic predictions with '
                             'argmax approach')

        return self._lotp_predict_proba(X)

    @staticmethod
    def _check_valid_clf(clf):
        """
        Checks that the provided classification model is a valid input

        Parameters
        ----------
        clf : str or object
            Classifier which we will be used in each parent node in the tree

        Returns
        -------
        None

        """
        if isinstance(clf, str):
            if not clf in ['knn', 'rf', 'logistic_regression']:
                raise ValueError('Only support "knn", "rf", and '
                                 '"logistic_regression" estimators')
        elif is_classifier(clf):
            # Check that clf has a predict_proba method because it is assumed
            # that the classifier can make probabilistic predictions
            if not hasattr(clf, 'predict_proba'):
                raise ValueError('Only support estimators which can make '
                                 'probabilistic predictions')
        else:
            raise TypeError('"clf" must either be a string or Scikit-learn'
                            'classifier; was given {}'.format(type(clf)))

        return None

    @staticmethod
    def _check_lda_compatible(orig_num_features, new_num_features):
        """
        Check if this is compatible to use LDA

        Parameters
        ----------
        orig_num_features : int
            Original number of features in the data
        new_num_features : int
            Desired number of features in transformed data

        Returns
        -------
        is_compatible : bool
            Whether LDA can be used for this data

        """
        if orig_num_features >= new_num_features:
            return True
        else:
            return False

    def _train_node(self, X, y, clf, **kwargs):
        """
        Trains one node in the HC graph

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        y : np.ndarray
            Label vector
        clf : str or object
            Classifier to use to train the node; if it is a string then
            the current supported values are {'knn', 'rf', and
            'logistic_regression'}; if it is an object then it is assumed
            that it is a Scikit-learn classification object that can make
            probabilistic predictions
        **kwargs
            Keyword arguments which define the hyper-parameters for the
            classification object


        Returns
        -------
        model : Pipeline
            Scikit-learn pipeline for the given node of the tree

        """

        if isinstance(clf, str):
            if clf is 'knn':
                clf = KNeighborsClassifier()
            elif clf is 'rf':
                clf = RandomForestClassifier(random_state=self.random_state)
            else:
                clf = SGDClassifier(loss='log', random_state=self.random_state)

        clf.set_params(**kwargs)

        scaler = StandardScaler()

        if self.num_features is None:
            # Default to num_labels - 1 for the desired number of features
            num_labels = len(np.unique(y))
            self.num_features = num_labels - 1
        elif not isinstance(self.num_features, int):
            raise TypeError('Desired number of features must be an integer')

        if self._check_lda_compatible(X.shape[1], self.num_features):
            features = LinearDiscriminantAnalysis(
                n_components=self.num_features
            )
        else:
            num_features = max(1, int(floor(X.shape[1] * self.feature_frac)))
            features = PCA(n_components=num_features,
                           random_state=self.random_state)

        steps = [('scaler', scaler), ('features', features), ('clf', clf)]
        pipe = Pipeline(steps)
        pipe.fit(X, y)
        return pipe

    def _check_valid_grouping(self):
        """
        Checks that the label grouping is valid

        Returns
        -------
        None

        """
        # A valid grouping is one where there are >= 2 meta-classes and less
        # than the total number of labels
        num_labels = len(self.label_groups)
        if self.num_metaclasses < 2:
            raise ValueError('Must have at least two meta-classes in the label '
                             'grouping')
        elif self.num_metaclasses >= num_labels:
            raise ValueError('Must have fewer meta-classes than the total '
                             'number of labels')
        else:
            return None

    def _remap_labels(self, y):
        """
        Maps the target vector according to the learned label grouping

        Parameters
        ----------
        y : np.ndarray[num_samples,]
            Label vector

        Returns
        -------
        y_new : np.ndarray[num_samples,]
            Re-mapped label vector

        """
        y_new = np.empty_like(y)
        for i in range(self.num_metaclasses):
            idx = np.isin(y, np.where(self.label_groups == i)[0])
            y_new[idx] = i
        return y_new

    def fit(self, X, y, clf, **kwargs):
        """
        Fits the hierarchical classifier to the provided data

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        y : np.ndarray
            Label vector
        clf : {'knn', 'rf', 'logistic_regression'} or object
            Classifier used to train the node; if it is an object then it is
            assumed that it is a Scikit-learn classification object that can
            make probabilistic predictions
        **kwargs
            Keyword arguments which define the hyper-parameters for the
            classification object

        Returns
        -------
        None

        """
        self._check_valid_clf(clf)

        # If the label groups have not yet been learned then this will be done
        # with the label grouping script
        if self.label_groups is None:
            self.label_groups = group_labels(
                X, y, self.group_algo, random_state=self.random_state,
                k=self.k, metric=self.metric, n_jobs=self.n_jobs,
                label_frac=self.label_frac
            )

            self.num_metaclasses = len(np.unique(self.label_groups))
            self.non_lone_leaves = self._get_leaf_idx()

        y_node = self._remap_labels(y)
        idx_list = [np.isin(y, np.where(self.label_groups == i)[0])
                    for i in self.non_lone_leaves]
        X_list = [X[idx] for idx in idx_list]
        y_list = [y[idx] for idx in idx_list]
        X_list.append(X)
        y_list.append(y_node)

        with Parallel(n_jobs=self.n_jobs) as p:
            models = p(delayed(self._train_node)(X, y, clf, **kwargs)
                       for (X, y) in zip(X_list, y_list))

        self.models = models
