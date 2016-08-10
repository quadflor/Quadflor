from unittest import TestCase
import numpy as np
from scipy.sparse import csr
from sklearn.neighbors import NearestNeighbors

from classifying.batch_kneighbors import BatchKNeighbors


class TestKNeighbors(TestCase):
    def test_inner_kneighbors(self):
        X = csr.csr_matrix([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        y = csr.csr_matrix([[0.4, 0.4, 0.4], [2.4, 2.4, 2.4], [3.1, 3.1, 3.1], [1.1, 1.1, 1.1]])
        nearest_neighbors = NearestNeighbors()
        nearest_neighbors.fit(X)
        neighbors = BatchKNeighbors(nearest_neighbors)

        kneighbors = neighbors._batch_kneighbors(y, n_neighbors=1, batchsize=1)
        np.testing.assert_array_equal(kneighbors, np.matrix([[0], [2], [3], [1]]))
        kneighbors = neighbors._batch_kneighbors(y, n_neighbors=1, batchsize=3)
        np.testing.assert_array_equal(kneighbors, np.matrix([[0], [2], [3], [1]]))
        kneighbors = neighbors._batch_kneighbors(y, n_neighbors=1, batchsize=10)
        np.testing.assert_array_equal(kneighbors, np.matrix([[0], [2], [3], [1]]))

    def test_inner_kneighbors_more_neighbors(self):
        X = csr.csr_matrix([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        y = csr.csr_matrix([[0.4, 0.4, 0.4], [2.4, 2.4, 2.4], [3.1, 3.1, 3.1], [1.1, 1.1, 1.1]])
        nearest_neighbors = NearestNeighbors()
        nearest_neighbors.fit(X)
        neighbors = BatchKNeighbors(nearest_neighbors)

        kneighbors = neighbors._batch_kneighbors(y, n_neighbors=2, batchsize=1)
        np.testing.assert_array_equal(kneighbors, np.matrix([[0, 1], [2,3], [3, 2], [1,2]]))

        kneighbors = neighbors._batch_kneighbors(y, n_neighbors=2, batchsize=3)
        np.testing.assert_array_equal(kneighbors, np.matrix([[0, 1], [2,3], [3, 2], [1,2]]))
