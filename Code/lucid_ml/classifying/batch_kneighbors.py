from timeit import default_timer

import datetime
import numpy as np
import sys


class BatchKNeighbors:
    """
    In sklearn, the whole distance matrix between the test data and the train data is
    calculated at once and stored in memory.
    Depending on the datatype of the matrix, this needs approximately
    num_test_samples * num_train_samples * 16 bytes of memory
    and for large data can be too large to keep in memory.
    When using this class instead, the neighbors are queried in batches.

    Parameters
    ----------
    knn: from sklearn.neighbors.NearestNeighbors or from sklearn.neighbors.LSHForest
        The nearest neighbor calculator.
    """

    def __init__(self, knn):
        self.knn = knn
        self.space_per_test_sample = 0

    def fit(self, X, y=None):
        """
        Fits the data to the knn.
        Parameters
        ----------
        X: {array-like, sparse matrix}
            Training data, shape [n_samples, n_features].
        y: ignored
        """
        self.space_per_test_sample = X.shape[0] * 16
        self.knn.fit(X)

    def _batch_kneighbors(self, X_test, n_neighbors, batchsize):
        num_samples = X_test.shape[0]
        indices = list(range(0, num_samples, batchsize)) + [num_samples]
        neighbors = np.empty((0, n_neighbors), dtype=np.int32)
        start = default_timer()
        for i0, i1 in zip(indices[:-1], indices[1:]):
            self._print_message(i0, i1, num_samples, start)
            batch = X_test[i0:i1]
            ns = self.knn.kneighbors(batch, n_neighbors=n_neighbors)[1]
            neighbors = np.vstack((neighbors, ns))
        return neighbors

    def kneighbors(self, X_test, n_neighbors=1, approx_max_ram_GB=10):
        """
        Calculates kneighbors in batches.
        Finds the K-neighbors of a point.

        Parameters
        ----------
        X_test: array-like, shape (n_query, n_features)
            The query point or points.
        n_neighbors: int, default = 1
            Number of neighbors to get.
        approx_max_ram_GB: int, default = 10
            Approxmiate maximum amount of RAM needed. Note that several aspects can influence this:
            Dataype or Parallelization in the knn passed to the constructer increases memory usage.

        Returns
        -------
        array
            Indices of the nearest points in the population matrix.
        """
        batchsize = int((approx_max_ram_GB * (2 ** 30)) / self.space_per_test_sample)
        return self._batch_kneighbors(X_test, n_neighbors, batchsize)

    def _print_message(self, i0, i1, num_samples, start):
        if i0 != 0:
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
        print('Neighbor batch ' + str(i0) + ' - ' + str(i1) + ' of ' + str(num_samples) + ' samples.' +
              ' ETA in: ' + self._calc_eta(i0, start, num_samples))

    @staticmethod
    def _calc_eta(i0, start, num_samples):
        if i0 == 0:
            return 'unknown'
        passed = default_timer() - start
        eta_s = int(num_samples * passed / i0 - passed)
        return str(datetime.timedelta(seconds=eta_s))
