from unittest import TestCase

import numpy as np
from scipy.sparse import csr
from sklearn.preprocessing import MultiLabelBinarizer

from lucid_ml.classifying.br_kneighbor_classifier import BRKNeighborsClassifier


class TestBRKNN(TestCase):
    def test_BRKnna_no_labels_take_closest(self):
        data = csr.csr_matrix([[0, 1], [1, 1], [1, 1.1], [0, 1]])
        train_ids = [['lid0', 'lid1'], ['lid2', 'lid3'], ['lid2', 'lid3'], ['lid0', 'lid5']]
        mlb = MultiLabelBinarizer(sparse_output=True)
        y = mlb.fit_transform(train_ids)
        knn = BRKNeighborsClassifier(n_neighbors=2, threshold=0.6, mode='a')
        knn.fit(data, y)

        pred = knn.predict(csr.csr_matrix([[0, 1]])).todense()
        print(pred)
        np.testing.assert_array_equal([[1, 0, 0, 0, 0]], pred)

    def test_BRKnna_predict(self):
        data = csr.csr_matrix([[0, 1], [1, 1], [1, 1.1], [0.5, 1]])
        train_ids = [['lid0', 'lid1'], ['lid2', 'lid3'], ['lid4', 'lid3'], ['lid4', 'lid5']]
        mlb = MultiLabelBinarizer(sparse_output=True)
        y = mlb.fit_transform(train_ids)

        knn = BRKNeighborsClassifier(threshold=0.5, n_neighbors=3, mode='a')
        knn.fit(data, y)

        pred = knn.predict(csr.csr_matrix([[1.1, 1.1]])).todense()
        np.testing.assert_array_equal([[0, 0, 0, 1, 1, 0]], pred)

    def test_BRKnna_predict_dense(self):
        data = csr.csr_matrix([[0, 1], [1, 1], [1, 1.1], [0.5, 1]])
        train_ids = [['lid0', 'lid1'], ['lid2', 'lid3'], ['lid4', 'lid3'], ['lid4', 'lid5']]
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(train_ids)

        knn = BRKNeighborsClassifier(threshold=0.5, n_neighbors=3, mode='a')
        knn.fit(data, y)

        pred = knn.predict(csr.csr_matrix([[1.1, 1.1]])).todense()
        np.testing.assert_array_equal([[0, 0, 0, 1, 1, 0]], pred)

    def test_BRKnnb_predict(self):
        data = csr.csr_matrix([[0, 1], [1, 1], [1.5, 1], [0.5, 1]])
        train_ids = [['lid0', 'lid1'], ['lid0', 'lid1'], ['lid4', 'lid3'], ['lid4', 'lid5']]
        mlb = MultiLabelBinarizer(sparse_output=True)
        y = mlb.fit_transform(train_ids)

        knn = BRKNeighborsClassifier(mode='b', n_neighbors=3)
        knn.fit(data, y)

        pred = knn.predict(csr.csr_matrix([[0, 1]])).todense()
        np.testing.assert_array_equal([[1, 1, 0, 0, 0]], pred)

    def test_BRKnnb_predict_dense(self):
        data = csr.csr_matrix([[0, 1], [1, 1], [1.5, 1], [0.5, 1]])
        train_ids = [['lid0', 'lid1'], ['lid0', 'lid1'], ['lid4', 'lid3'], ['lid4', 'lid5']]
        mlb = MultiLabelBinarizer(sparse_output=False)
        y = mlb.fit_transform(train_ids)

        knn = BRKNeighborsClassifier(mode='b', n_neighbors=3)
        knn.fit(data, y)

        pred = knn.predict(csr.csr_matrix([[0, 1]])).todense()
        np.testing.assert_array_equal([[1, 1, 0, 0, 0]], pred)

    def test_BRKnnb_predict_two_samples(self):
        data = csr.csr_matrix([[0, 1], [1, 1.1], [1, 1], [0.5, 1]])
        train_ids = [['lid0', 'lid1'], ['lid0', 'lid1'], ['lid4', 'lid5'], ['lid4', 'lid5']]
        mlb = MultiLabelBinarizer(sparse_output=True)
        y = mlb.fit_transform(train_ids)

        knn = BRKNeighborsClassifier(mode='b', n_neighbors=3)
        knn.fit(data, y)

        pred = knn.predict(csr.csr_matrix([[0, 1], [2, 2]])).todense()
        np.testing.assert_array_equal([[1, 1, 0, 0], [0, 0, 1, 1]], pred)

    def test_BRKnnb_auto_optimize_k(self):
        data = csr.csr_matrix([[0, 1], [1, 1], [0, 1.1], [1.1, 1]])
        train_ids = [['lid0', 'lid1'], ['lid0', 'lid1'], ['lid2', 'lid3'], ['lid0', 'lid1']]
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(train_ids)

        knn = BRKNeighborsClassifier(mode='b', n_neighbor_candidates=[1, 3], auto_optimize_k=True)

        # noinspection PyUnusedLocal
        def fun(s, X, y_):
            return data[[1, 2, 3]], data[[0]], y[[1, 2, 3]], y[[0]]

        BRKNeighborsClassifier._get_split = fun
        knn.fit(data, y)
        self.assertEquals(3, knn.n_neighbors)
        pred = knn.predict(csr.csr_matrix([[0.1, 1], [2, 2]])).todense()
        np.testing.assert_array_equal([[1, 1, 0, 0], [1, 1, 0, 0]], pred)

        # def test_time_brknnb(self):
        #     times = []
        #     X = sp.rand(10000, 5000, density=0.005, format='csr')
        #     y = sp.rand(10000, 3000, density=0.005, format='csr')
        #     knn = BRKNeighborsClassifier(n_neighbors=100)
        #     knn.fit(X,y)
        #     X_test = sp.rand(1000, 5000, density=0.005, format ='csr')
        #     for _ in range(5):
        #         start = default_timer()
        #         knn.predict(X_test)
        #         times.append(default_timer() - start)
        #     print(np.mean(times))
