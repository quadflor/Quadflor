import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator

class MeanCutShell(BaseEstimator):
    '''An Ensemble method for large scale multi-label classification problems
    with many classes but only few labels per document.
    '''
    def __init__(self, *clfs, soft=False):
        self.clfs = clfs
        print(type(self.clfs))
        self.soft=soft
        self.verbose = 1

    def fit(self, X, Y):
        self.n_topics = Y.shape[1]
        ones = len(sp.find(Y)[2])
        self.mu = ones / X.shape[0]
        for clf in self.clfs:
            clf.fit(X, Y)
        return self

    def partial_fit(self, X, Y):
        for clf in self.clfs:
            clf.partial_fit(X, Y)
        return self

    def predict(self, X):
        n_query = X.shape[0]
        ''' Compute Y_probas
        Y_probas shape should be (n_query, n_topics * len(clfs))
        '''
        Y_probas = np.hstack(clf.predict_proba(X) for clf in self.clfs)
        print("Y_probas shape:", Y_probas.shape)
        mean_cut = self.mu
        if self.verbose:
            print("MeanCut = %d" % mean_cut)
        Y_pred = sp.csr_matrix((0, self.n_topics))
        for i in range(n_query): # sample based predictions, can probably be avoided
            indices = np.argsort(Y_probas[i])[-mean_cut:] # sort probabilites row-wise
            data = np.array([1] * len(indices))
            indptr = np.array([0, 1])
            row = sp.csr_matrix((data, indices, indptr), shape=(1, self.n_topics))
            Y_pred = sp.vstack((Y_pred, row))

        return Y_pred
