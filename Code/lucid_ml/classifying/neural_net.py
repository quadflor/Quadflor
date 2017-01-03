#!/usr/bin/env python3
# coding: utf-8
from scipy import sparse

from sklearn.base import BaseEstimator
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import Ridge


def _batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        y_batch = y[batch_index].toarray()
        counter += 1
        yield X_batch, y_batch
        if counter == number_of_batches:
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def _batch_generatorp(X, batch_size):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if counter == number_of_batches:
            counter = 0


class MLP(BaseEstimator):
    def __init__(self, verbose=0, model=None, final_activation='sigmoid'):
        self.verbose = verbose
        self.model = model
        self.final_activation = final_activation

    def fit(self, X, y):
        if not self.model:
            self.model = Sequential()
            self.model.add(Dense(1000, input_dim=X.shape[1]))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(y.shape[1]))
            self.model.add(Activation(self.final_activation))
            self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
        self.model.fit_generator(generator=_batch_generator(X, y, 256, True),
                                 samples_per_epoch=X.shape[0], nb_epoch=20, verbose=self.verbose)

    def predict(self, X):
        pred = self.predict_proba(X)
        return sparse.csr_matrix(pred > 0.2)

    def predict_proba(self, X):
        pred = self.model.predict_generator(generator=_batch_generatorp(X, 512), val_samples=X.shape[0])
        return pred


class ThresholdingPredictor(BaseEstimator):
    """
    Class for the thresholding predictor which wraps a probabilistic model for multi label classification
    >>> mlp = MLP()
    >>> tp = ThresholdingPredictor(mlp, alpha=1.0, stepsize=0.01, verbose=0)
    >>> X = np.random.randn(100, 42)
    >>> X = sparse.csr_matrix(X)
    >>> Y = sparse.csr_matrix(np.random.rand(100,6) > .5)
    >>> X.shape
    (100, 42)
    >>> Y.shape
    (100, 6)
    >>> _ = tp.fit(X,Y)
    >>> f1_score(Y, tp.predict(X), average='samples') > 0.5
    True
    """
    def __init__(self,
                 probabilistic_estimator,
                 metric_average='samples',
                 stepsize=0.01,
                 verbose=0,
                 fit_intercept=False,
                 sparse_output=True,
                 **ridge_params
                 ):
        """
        Arguments:
            probabilistic_estimator -- Estimator capable of predict_proba

        Keyword Arguments:
            average -- averaging method for f1 score
            stepsize -- stepsize for the exhaustive search of optimal threshold
            fit_intercept -- fit intercept in Ridge regression
            sparse_output -- Predict returns csr in favor of ndarray
            **ridge_params -- Passed down to Ridge regression
        """
        self.model = probabilistic_estimator
        self.verbose = verbose
        self.metric_average = metric_average
        self.ridge = Ridge(fit_intercept=fit_intercept, **ridge_params)
        self.stepsize = stepsize
        self.sparse_output = sparse_output

    def fit(self, X, y):
        """
        Arguments:
            X -- ndarray [n_samples, n_features]
            y -- label indicator matrix [n_samples, n_outputs]
        """
        model, ridge = self.model, self.ridge
        avg, step = self.metric_average, self.stepsize

        # Fit probabilistic model
        model.fit(X, y)

        # let it predict the probablities
        probas = model.predict_proba(X)

        # exhaustive search for optimal threshold
        ts = np.arange(0.0, 1.0, step)
        if self.verbose > 0:
            print("[TP] Exhaustive search for optimal threshold...", end=' ')
        f1s = np.asarray([f1_score(y, probas >= t, average=avg) for t in ts])
        t_opt = ts[np.argmax(f1s)]
        if self.verbose > 0:
            print("t_opt = {}".format(t_opt))

        # linear regression from inputs to optimal threshold
        if self.verbose > 0:
            print("[TP] Fitting ridge regression...", end=' ')
        T = np.full((X.shape[0], 1), t_opt)
        ridge.fit(X, T)
        if self.verbose > 0:
            print("Done.")

        return self

    def predict(self, X):
        """
        Arguments:
            X -- ndarray, csr_matrix [n_samples, n_features]
        Returns:
            Predictions as label indicator matrix (sparse)
        """
        model, ridge, verbose = self.model, self.ridge, self.verbose
        pred = model.predict_proba(X)

        threshold = ridge.predict(X)
        if self.verbose:
            print("Inferred threshold t = {}".format(threshold))

        labels = pred >= threshold

        if self.sparse_output:
            return sparse.csr_matrix(labels)
        else:
            return labels


if __name__ == "__main__":
    import doctest
    doctest.testmod()
