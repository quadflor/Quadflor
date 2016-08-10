from timeit import default_timer

import datetime
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.neighbors import LSHForest, NearestNeighbors

from classifying.batch_kneighbors import BatchKNeighbors


class NearestNeighbor(BaseEstimator):
    """
        Simple classifier, which just takes the single nearest neighbor.
        For some reason using sklearn.neighbors.NearestNeighbors is significantly faster
        than using sklearn.neighbors.KNearestNeighborClassifier.

        Parameters
        ----------
        use_lsh_forest: bool, default = False
            Use approximate nearest neighbor.
        metric: str, default = 'cosine'
            The metric.
        algorithm: str, default = 'brute'
            The algorithm.
    """
    def __init__(self, use_lsh_forest=False, metric='cosine', algorithm='brute'):
        self.lsh = use_lsh_forest
        self.y = None
        nn = LSHForest(n_neighbors=1, n_candidates=400, n_estimators=35) if use_lsh_forest else NearestNeighbors(
                n_neighbors=1, metric=metric, algorithm=algorithm)
        self.knn = BatchKNeighbors(nn)

    def fit(self, X, y=None):
        """
        Fit the model using X as training data and y as target values.
        Parameters
        ----------
        X: {array-like, sparse matrix}
            Training data, shape [n_samples, n_features].
        y: {array-like, sparse matrix}
            Target values of shape = [n_samples] or [n_samples, n_outputs]
        """
        self.y = y
        self.knn.fit(X, y)

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X: array-like, shape (n_query, n_features), or (n_query, n_indexed) if metric == ‘precomputed’
            Test samples.

        Returns
        -------
        array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        start = default_timer()
        neighbor_ids = self.knn.kneighbors(X)
        result = sp.csr_matrix((0, self.y.shape[1]))
        for n in neighbor_ids:
            neighbor_labels = self.y[n]
            result = sp.vstack((result, neighbor_labels))

        print('Prediction took ' + str(datetime.timedelta(seconds=default_timer() - start)))
        return result
