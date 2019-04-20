import warnings
import numpy as np
import pytest
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ..gen_hc import GenHC


def test_check_valid_clf():
    hc = GenHC()
    with pytest.raises(ValueError):
        hc._check_valid_clf('svm')

    with pytest.raises(ValueError):
        hc._check_valid_clf(LinearSVC())

    with pytest.raises(TypeError):
        hc._check_valid_clf(1)


def test_train_node():
    # The LDA might give a warning about being collinear, but I do not care
    # for testing purposes
    warnings.filterwarnings("ignore", category=UserWarning)

    # Check that we can add hyper-parameters for a variety of estimators
    X, y = make_classification(random_state=17)
    hc = GenHC()
    model = hc._train_node(X, y, clf='rf', n_estimators=5)
    assert len(model.named_steps['clf'].estimators_) == 5

    model = hc._train_node(X, y, clf='knn', n_neighbors=5)
    assert model.named_steps['clf'].n_neighbors == 5

    # Check that we reach an LDA state when there are enough features and
    # PCA when there are not
    assert isinstance(model.named_steps['features'], LinearDiscriminantAnalysis)

    X, y = make_classification(n_features=5, n_informative=5, n_redundant=0,
                               n_repeated=0, n_classes=7, random_state=17)
    hc = GenHC()
    model = hc._train_node(X, y, clf='rf', n_estimators=5)
    assert isinstance(model.named_steps['features'], PCA)


def test_fit():
    warnings.filterwarnings('ignore', category=UserWarning)

    X, y = make_classification(n_classes=3, n_features=10, n_informative=10,
                               n_redundant=0, n_repeated=0,
                               n_clusters_per_class=1, random_state=17)
    hc = GenHC(k=2, random_state=17)
    hc.fit(X, y, clf='rf', n_estimators=10)

    # Check that two models have been fit
    assert len(hc.models) == 2

    # Check that multiprocessing version works
    hc = GenHC(k=2, random_state=17, n_jobs=2)
    hc.fit(X, y, clf='rf', n_estimators=10)
    assert len(hc.models) == 2


def test_proba_predict():
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    X, y = make_classification(n_classes=3, n_features=10, n_informative=10,
                               n_redundant=0, n_repeated=0,
                               n_clusters_per_class=1, random_state=17)

    hc = GenHC(k=2, random_state=17)
    hc.fit(X, y, clf='rf', n_estimators=10)
    Y_pred = hc.predict_proba(X)

    # Every row should sum to one
    num_samples = Y_pred.shape[0]
    assert np.allclose(Y_pred.sum(axis=1), np.ones(shape=(num_samples,)))
    assert np.isnan(Y_pred).sum() == 0

    # Check for stable outputs from all supported models
    hc = GenHC(k=2, random_state=17)
    hc.fit(X, y, clf='knn')
    Y_pred = hc.predict_proba(X)
    assert np.isnan(Y_pred).sum() == 0

    hc = GenHC(k=2, random_state=17)
    hc.fit(X, y, clf='logistic_regression')
    Y_pred = hc.predict_proba(X)
    assert np.isnan(Y_pred).sum() == 0


def test_predict():
    warnings.filterwarnings('ignore', category=UserWarning)
    X, y = make_classification(n_samples=10, n_classes=3, n_features=10,
                               n_informative=10, n_redundant=0, n_repeated=0,
                               n_clusters_per_class=1, random_state=17)

    hc = GenHC(k=2, random_state=17)
    hc.fit(X, y, clf='rf', n_estimators=10)
    y_pred = hc.predict(X)
    assert np.allclose(np.unique(y_pred), np.array([0, 1, 2]))

    hc = GenHC(k=2, random_state=17, prediction_method='argmax')
    hc.fit(X, y, clf='rf', n_estimators=10)
    y_pred = hc.predict(X)
    assert np.allclose(np.unique(y_pred), np.array([0, 1, 2]))

    # Check that the example is correct
    hc = GenHC()
    hc.fit(X, y, clf='rf', n_estimators=10)
    y_pred = hc.predict(X)
    print(y_pred)
    assert np.allclose(y_pred, np.array([2, 0, 0, 1, 0, 0, 0, 2, 1, 2]))


def test_node_predict_proba():
    warnings.filterwarnings('ignore', category=UserWarning)
    X, y = make_classification(n_classes=3, n_features=10, n_informative=10,
                               n_redundant=0, n_repeated=0,
                               n_clusters_per_class=1, random_state=17)

    hc = GenHC(k=2, random_state=17)
    hc.fit(X, y, clf='rf', n_estimators=10)
    y_pred = hc.node_predict_proba(X).argmax(axis=1)
    assert np.allclose(np.unique(y_pred), np.array([0, 1]))
