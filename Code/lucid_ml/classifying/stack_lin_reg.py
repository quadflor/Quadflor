from scipy import sparse

from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
import numpy as np


class LinRegStack(BaseEstimator):
    def __init__(self, nn, verbose=0, fit_base=True):
        self.fit_base = fit_base
        self.verbose = verbose
        self.nn = nn
        self.linreg = Ridge()

    def fit(self, X, y):
        if self.fit_base:
            self.nn.fit(X, y)
        if self.verbose:
            print('Predicting train data with base classifier...')
        preds = self.nn.predict_proba(X)

        if self.verbose:
            print('Calculating thresholds...')
            print(X.shape, y.shape, preds.shape)
        i = 0
        training_thresholds = []
        for y_row, probs in zip(y, preds):
            labels = np.nonzero(y_row)
            if self.verbose and i % 100 == 0:
                print('\r', end='')
                print(str(i) + ' of ' + str(X.shape[0]), end='')

            i += 1
            f1 = -1
            threshold = None
            true_preds = preds[labels]
            # true_preds = list(sorted(true_preds))

            # for p0, p1 in zip(true_preds, true_preds[1:] + [0]):
            for p in true_preds:
                # p = p1 + (p0 - p1) / 2
                score = f1_score(y_row.toarray().T, probs >= p)
                if score > f1:
                    f1 = score
                    threshold = p
            training_thresholds.append(threshold)
        print('Mean threshold: ' + str(np.mean(training_thresholds)))

        if self.verbose:
            print('Fitting ridge regression..')
            print(training_thresholds[:5])
        self.linreg.fit(X, training_thresholds)

    def predict(self, X):
        thresholds = self.linreg.predict(X)
        preds = self.nn.predict_proba(X)
        return sparse.csr_matrix(preds >= thresholds[:, None])
