import sys
from unittest import TestCase

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from lucid_ml.weighting.statistical.bm25transformer import BM25Transformer

sys.path.append('../..')


class TestBM25(TestCase):
    def setUp(self):
        docs = [('id0', 'Lorem ipsum'), ('id1', 'Lorem Lorem ipsum dolor sit amet')]
        countvectorizer = CountVectorizer()
        self.frequency_matrix = countvectorizer.fit_transform([i[1] for i in docs])
        self.bm = BM25Transformer(0.75, 1.6)

    def testNumberOfEntities(self):
        expected_values = np.array((2, 5))
        results = self.bm._number_of_entities(self.frequency_matrix)
        self.assertTrue((results == expected_values).all())

    def testAvgdl(self):
        self.assertEquals(3.5, self.bm._average_number_of_entities(self.frequency_matrix))

    def testBm25_term(self):
        # alphabetic: amet dolor ipsum lorem sit
        expected_matrix = np.array([[0, 0, 1.24658, 1.24658, 0],
                                    [0.8349, 0.8349, 0.8349, 1.2639, 0.8349]])
        result_matrix = self.bm._bm_term(self.frequency_matrix)
        np.testing.assert_array_almost_equal(expected_matrix, result_matrix, decimal=4)

    def test_idf_bm25(self):
        frequency_matrix = self.frequency_matrix
        # alphabetic: amet dolor ipsum lorem sit
        expected_matrix = np.array([np.log(1), np.log(1), np.log(0.2), np.log(0.2), np.log(1)])
        idf_values = self.bm._idf_bm25(frequency_matrix)
        np.testing.assert_array_almost_equal(expected_matrix, idf_values)
