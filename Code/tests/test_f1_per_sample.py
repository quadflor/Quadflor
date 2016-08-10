from unittest import TestCase

import numpy as np
from scipy.sparse import csr

from utils.metrics import f1_per_sample


class TestF1PerSample(TestCase):
    def test_simple(self):
        y_true = csr.csr_matrix([[1, 0], [1, 0], [1, 0], [1, 1]])
        y_pred = csr.csr_matrix([[1, 0], [0, 1], [1, 1], [0, 1]])

        np.testing.assert_array_equal(f1_per_sample(y_true, y_pred), [1.,0., 2/3, 2/3])

    def test_simple_dense(self):
        y_true = np.matrix([[1, 0], [1, 0]])
        y_pred = np.matrix([[1, 0], [0, 1]])

        np.testing.assert_array_equal(f1_per_sample(y_true, y_pred), [1.,0.])
