from scipy import sparse

from sklearn.base import BaseEstimator
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np


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
    def __init__(self, verbose=0, model=None):
        self.verbose = verbose
        self.model = model

    def fit(self, X, y):
        if not self.model:
            self.model = Sequential()
            self.model.add(Dense(1000, input_dim=X.shape[1]))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(y.shape[1]))
            self.model.add(Activation('sigmoid'))
            self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
        self.model.fit_generator(generator=_batch_generator(X, y, 256, True),
                                 samples_per_epoch=X.shape[0], nb_epoch=20, verbose=self.verbose)

    def predict(self, X):
        pred = self.predict_proba(X)
        return sparse.csr_matrix(pred > 0.2)

    def predict_proba(self, X):
        pred = self.model.predict_generator(generator=_batch_generatorp(X, 512), val_samples=X.shape[0])
        return pred
