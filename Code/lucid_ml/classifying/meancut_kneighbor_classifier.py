""" Probabilistic KNeighbors Classification """
# -*- coding: utf-8 -*-
import numpy as np
import random
from scipy.sparse import lil_matrix, csr_matrix, vstack, find

from sklearn.neighbors.base import NeighborsBase, KNeighborsMixin, SupervisedIntegerMixin
from sklearn.base import ClassifierMixin
from sklearn.utils import check_array
from math import sqrt

class MeanCutKNeighborsClassifier(NeighborsBase, KNeighborsMixin, ClassifierMixin):
    def __init__(self, n_neighbors=5,
            soft=False, algorithm='auto', leaf_size=30,
            p=2, metric='minkowski', metric_params=None, n_jobs=1,
            **kwargs):
        self._init_params(n_neighbors=n_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size, metric=metric, p=p,
                metric_params=metric_params, n_jobs=n_jobs, **kwargs)
        self.soft = soft

    def fit(self, X, y=None):
        self._y = y

        n_samples = X.shape[0]

        ## compute mean of labels per sample
        mu = len(find(y)[0])
        mu /= n_samples

        #store mu
        self.mu = mu

        ## compute std deviation of labels per sample
        if self.soft:
            sigma = float(0)
            for row in self._y:
                d = len(find(row)[0])
                sigma += (d - mu) ** 2
            sigma /= (n_samples - 1)
            sigma = sqrt(sigma)
            # store sigma
            self.sigma = sigma

        return self._fit(X)


    def predict(self, X):
        X = check_array(X, accept_sparse='csr')
        neigh_ind = self.kneighbors(X, return_distance=False)

        #classes_ = self.classes_
        _y = self._y
        mu = self.mu
        if self.soft: sigma = self.sigma

        #n_outputs = len(classes_)
        n_samples = X.shape[0]
        n_outputs = _y.shape[1]

        mean_cut = int(round(mu))
        '''DEBUG OUTPUT
        if self.soft:
            print("MeanCut: %.2f (+/- %.2f)"%(mu,2*sigma))
        else:
            print("MeanCut: %d"%mean_cut)
        '''
        y_pred = csr_matrix((0, n_outputs), dtype=np.int64)
        for sample in neigh_ind:
            if self.soft: mean_cut = max(1, int(round(np.random.normal(mu, sigma))))
            neighbor_labels = _y[sample]
            label_sums = neighbor_labels.getnnz(0)
            label_indices = np.argsort(label_sums)[-mean_cut:]
            row = np.zeros((1,n_outputs),dtype=np.int64)
            row[0,label_indices] = 1
            row = csr_matrix(row)
            y_pred = vstack((y_pred, row))

        return y_pred



        
