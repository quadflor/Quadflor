from collections import defaultdict

import multiprocessing

import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.base import BaseEstimator
from sklearn.externals.joblib import Parallel, delayed
from sklearn.tree.tree import DecisionTreeClassifier

from classifying.rocchioclassifier import RocchioClassifier


class ClassifierStack(BaseEstimator):
    def __init__(self, n=50, base_classifier=RocchioClassifier(), n_jobs=1):
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self.meta_classifiers = {}
        self.n = n
        self.base_classifier = base_classifier

    def fit(self, X, y):
        """
        Fit the NearestCentroid model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
            Note that centroid shrinking cannot be used with sparse matrices.
        y : array, shape = [n_samples]
            Target values (integers)
        """

        self.y = y

        self.base_classifier.fit(X, y)
        distances = self.base_classifier.predict_proba(X)
        topNIndices, topNDistances = self._get_top_labels(distances)
        training_data = self._extract_features(topNIndices, topNDistances, y)

        # create a decision tree for each label
        self.meta_classifiers = {}
        for label, training_samples_of_label in training_data.items():
            training_samples_of_label = np.matrix(training_samples_of_label)
            decision_tree = DecisionTreeClassifier(criterion="gini")
            decision_tree.fit(training_samples_of_label[:, 0:2], training_samples_of_label[:, 2])
            self.meta_classifiers[label] = decision_tree

    def _get_top_labels(self, probas):
        topNIndices = np.apply_along_axis(lambda x: list(np.argsort(x)[-self.n:]), 1, probas)
        topNDistances = np.apply_along_axis(lambda x: list(np.sort(x)[-self.n:]), 1, probas)
        return [topNIndices, topNDistances]

    def _extract_features(self, topNIndices, topNDistances, y):
        samples = self._split_samples(topNIndices, topNDistances, y)
        training_data_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_analyze)(tI, tD, y) for tI, tD, y in samples)

        # merge training data
        training_data = defaultdict(list)
        for training_data_dict in training_data_list:
            for label, training_samples_of_label in training_data_dict.items():
                training_data[label].extend(training_data_dict[label])
        return training_data

    def _split_samples(self, topNIndices, topNDistances, y = None):
        num_sets = self.n_jobs

        set_size = int(len(topNIndices) / num_sets)
        sub_sets = [(topNIndices[i * set_size:(i + 1) * set_size], topNDistances[i * set_size:(i + 1) * set_size],
                     y[i * set_size:(i + 1) * set_size, :]) for i in range(num_sets - 1)]
        sub_sets = sub_sets + [(topNIndices[(num_sets - 1) * set_size :],
                                topNDistances[(num_sets - 1) * set_size :], y[(num_sets - 1) * set_size : , :])]
        return sub_sets

    def predict(self, X):

        predictions = dok_matrix((X.shape[0], self.y.shape[1]), dtype=np.int)

        topNIndices, topNDistances = self._get_top_labels(self.base_classifier.predict_proba(X))

        for entry, (label_list, dist_list) in enumerate(zip(topNIndices, topNDistances)):
            for rank, label in enumerate(label_list):
                training_sample = [[rank, dist_list[rank]]]
                if label in self.meta_classifiers:
                    prediction = self.meta_classifiers[label].predict(training_sample)[0]
                    if prediction == 1:
                        predictions[entry, label] = 1

        return csr_matrix(predictions)


def _analyze(topNIndices, topNDistances, y):
    training_data = defaultdict(list)
    for entry, (label_list, dist_list) in enumerate(zip(topNIndices, topNDistances)):
        for rank, (label, dist) in enumerate(zip(label_list, dist_list)):
            training_data[label].append([rank, dist, 1 if y[entry, label] == 1 else 0])
    return training_data


