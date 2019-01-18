import copy
import datetime
import pprint
from timeit import default_timer

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics.scorer import check_scoring
from sklearn.neighbors import NearestNeighbors, LSHForest

from classifying.batch_kneighbors import BatchKNeighbors


class BRKNeighborsClassifier(BaseEstimator):
    """
    The Binary Relevance K-Neighbors Classifier has two modes:

    Mode a:

    Take the k nearest neighbors and vote for each label. If a labels relative frequency in the neighbors
     is below the given threshold, it is discarded. If no labels remain, the label with the highest relative
     occurrence is taken.

    Mode b:

    Take the k nearest neighbors and vote for each label. Sort the list by relative frequency and take
    the top n labels, where n is the floored average number of labels taken from all k neighbors.

    Parameters
    ----------
    threshold : float, default = 0.2
        Threshold for mode a
    use_lsh_forest : bool, default = False
        Use approximate k-nearest neighbor
    mode : str, default = 'b'
        'a' or 'b' to choose mode.
    n_neighbors : int, default = 50
        Number of nearest neighbors when not automatically optimizing k.
    scoring : str or Callable, default = 'f1_samples'
        Scoring Function for optimizing k. Greate is better.
    auto_optimize_k: bool, default = False
        Automatically tries all n_neighbor_candidates and compares scoring on validation set.
    n_neighbor_candidates: List[int], default = (3, 5, 8, 13, 21, 34, 55, 84, 139, 223, 362)
        Candidates for automatic k-Optimization
    algorithm: str, default='brute'
        Algorithm to pass to sklearn.neighbors.NearestNeighbors
    metric: str, default='cosine
        Metric to pass to sklearn.neighbors.NearestNeighbors
    """

    def __init__(self, threshold=0.2, use_lsh_forest=False, mode='b',
                 n_neighbors=50, scoring='f1_samples', auto_optimize_k=False,
                 n_neighbor_candidates=(3, 5, 8, 13, 21, 34, 55, 84, 139, 223, 362),
                 algorithm='brute', metric='cosine'):
        self.auto_optimize_k = auto_optimize_k
        self.scoring = scoring
        self.n_neighbor_candidates = n_neighbor_candidates
        self.mode = mode
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        nn = LSHForest(n_neighbors=n_neighbors, n_candidates=400,
                       n_estimators=35) if use_lsh_forest else NearestNeighbors(
                n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        self.knn = BatchKNeighbors(nn)
        self.y = None

    def fit(self, X, y):
        """ Fit the model using X as training data and y as target values.
        If auto_optimize_k is True, searches n_neighbor_candidates for best k on validation set of size 0.1 *
        X.shape[0].

        Parameters
        ----------
        X: {array-like, sparse matrix}
            Training data, shape [n_samples, n_features].
        y: {array-like, sparse matrix}
            Target values of shape = [n_samples] or [n_samples, n_outputs]

        Returns
        -------
        BRKNeighborsClassifier
            Estimator fit to the data.

        """
        if self.auto_optimize_k:
            self._optimize_n_neighbors(X, y)
        self.y = y
        self.knn.fit(X)
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape (n_query, n_features)
            Test data.

        Returns
        -------
        array of shape = [n_samples, n_classes]
                The predicted labels.

        """
        start = default_timer()
        neighbor_ids = self.knn.kneighbors(X, n_neighbors=self.n_neighbors)
        prediction = self._a(neighbor_ids) if self.mode == 'a' else self._b(neighbor_ids)
        print('Prediction took ' + str(datetime.timedelta(seconds=default_timer() - start)))
        return prediction

    def _a(self, neighbor_ids):
        result = sp.csr_matrix((0, self.y.shape[1]))
        for ns in neighbor_ids:
            neighbor_labels = self.y[ns]
            # By squeezing we support matrix output from scipy.sparse.sum and 1D array from np.sum
            labels_sum = np.squeeze(np.array(neighbor_labels.sum(0)))
            predicted_labels = sp.csr_matrix([np.floor(np.divide(labels_sum, len(ns)) + (1 - self.threshold))])
            # If there are no labels, we take the most frequent label.
            if predicted_labels.sum() == 0:
                divide = np.divide(labels_sum, len(ns))
                max_label = divide.argmax()
                predicted_labels = sp.dok_matrix((1, predicted_labels.shape[1]))
                predicted_labels[0, max_label] = 1
                predicted_labels = sp.csr_matrix(predicted_labels)

            result = sp.vstack((result, predicted_labels))
        return result

    def _b(self, neighbor_ids):
        result = sp.csr_matrix((0, self.y.shape[1]))
        for ns in neighbor_ids:
            average_label_nums = int(np.floor(np.mean([self.y[n].sum() for n in ns])))
            neighbor_labels = self.y[ns]
            labels_sum = np.array(neighbor_labels.sum(0))
            # By squeezing we support matrix output from scipy.sparse.sum and 1D array from np.sum
            divide = np.squeeze(np.divide(labels_sum, len(ns)))
            predicted_indices = np.argsort(divide)[-average_label_nums:]
            predicted_labels = sp.dok_matrix((1, len(divide)))
            # noinspection PyTypeChecker
            for index in predicted_indices:
                predicted_labels[0, index] = 1
            predicted_labels = sp.csr_matrix(predicted_labels)
            result = sp.vstack((result, predicted_labels))
        return result

    def _optimize_n_neighbors(self, X, y):
        print('Auto optimizing n_neighbors using ' + str(self.n_neighbor_candidates))
        X_train, X_validate, y_train, y_validate = self._get_split(X, y)
        estimator = copy.copy(self)
        estimator.auto_optimize_k = False
        estimator.fit(X_train, y_train)
        scorer = check_scoring(estimator, scoring=self.scoring)
        configs = []
        for n_neighbors in self.n_neighbor_candidates:
            estimator.n_neighbors = n_neighbors
            score = scorer(estimator, X_validate, y_validate)
            print('N_neighbors = ' + str(n_neighbors) + ' score: ' + str(self.scoring) + ' ' + str(score))
            configs.append((n_neighbors, score))

        configs = sorted(configs, key=lambda i: i[1], reverse=True)
        print('Configs in order of score: ')
        pprint.pprint(configs)
        self.n_neighbors = configs[0][0]

    @staticmethod
    def _get_split(X, y):
        split = ShuffleSplit(y.shape[0], n_iter=1)
        train, validate = list(split)[0]
        X_train, X_validate, y_train, y_validate = X[train], X[validate], y[train], y[validate]
        return X_train, X_validate, y_train, y_validate
