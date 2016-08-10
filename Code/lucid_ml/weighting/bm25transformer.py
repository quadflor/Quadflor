# -*- coding: utf-8 -*-
# Authors: Olivier Grisel <olivier.grisel@ensta.org>
# Mathieu Blondel <mathieu@mblondel.org>
# Lars Buitinck <L.J.Buitinck@uva.nl>
# Robert Layton <robertlayton@gmail.com>
#          Jochen Wersdörfer <jochen@wersdoerfer.de>
#          Roman Sinayev <roman.sinayev@gmail.com>
#
# License: BSD 3 clause

from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import normalize


class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Transforms a document-term (count-)matrix into a term-document matrix weighted according to BM25.
    
    Parameters
    ----------
    b : float, default=0.75
        The b parameter from BM25. Should be between zero and one.
    k : float, default=1.6
        The k parameter from BM25.
    """

    def __init__(self, norm='l2', use_idf=True, use_bm25idf=False, smooth_idf=True,
                 delta_idf=False, sublinear_tf=False, bm25_tf=False, k=1.6, b=0.75):
        self.norm = norm
        self.use_idf = use_idf
        self.use_bm25idf = use_bm25idf
        self.smooth_idf = smooth_idf
        # Required for delta idf's
        self.delta_idf = delta_idf

        self.sublinear_tf = sublinear_tf
        self.bm25_tf = bm25_tf
        self.k = k
        self.b = b

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights).

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)

        if self.use_idf:
            n_samples, n_features = X.shape

            # BM25 idf
            if self.use_bm25idf:            
                if self.delta_idf:
                    if y is None:
                        raise ValueError("Labels are needed to determine Delta idf")

                    N1, df1, N2, df2 = _class_frequencies(X, y)
                    delta_bm25idf = np.log(((N1 - df1 + 0.5) * df2 + 0.5) / ((N2 - df2 + 0.5) * df1 + 0.5))
                    self._idf_diag = sp.spdiags(delta_bm25idf,
                                            diags=0, m=n_features, n=n_features)
                else:
                    # vanilla bm25 idf
                    df = _document_frequency(X)

                    # perform idf smoothing if required
                    df += int(self.smooth_idf)
                    n_samples += int(self.smooth_idf)

                    # log1p instead of log makes sure terms with zero idf don't get
                    # suppressed entirely

                    bm25idf = np.log((n_samples - df + 0.5) / (df + 0.5))
                    self._idf_diag = sp.spdiags(bm25idf,
                                                diags=0, m=n_features, n=n_features)

            # Vanilla idf
            elif self.delta_idf:
                if y is None:
                    raise ValueError("Labels are needed to determine Delta idf")

                N1, df1, N2, df2 = _class_frequencies(X, y)
                delta_idf = np.log((df1 * float(N2) + int(self.smooth_idf)) /
                                   (df2 * N1 + int(self.smooth_idf)))

                # Maybe scale delta_idf to only positive values (for Naive Bayes, etc) ?
                self._idf_diag = sp.spdiags(delta_idf,
                                            diags=0, m=n_features, n=n_features)

            else:
                df = _document_frequency(X)

                # perform idf smoothing if required
                df += int(self.smooth_idf)
                n_samples += int(self.smooth_idf)

                # log1p instead of log makes sure terms with zero idf don't get
                # suppressed entirely
                idf = np.log(float(n_samples) / df) + 1.0
                self._idf_diag = sp.spdiags(idf,
                                            diags=0, m=n_features, n=n_features)
       

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.bm25_tf:

            # First calculate the denominator of BM25 equation
            # Sum the frequencies (sum of each row) to get the documents lengths
            D = (X.sum(1) / np.average(X.sum(1))).reshape((n_samples, 1))
            D = ((1 - self.b) + self.b * D) * self.k
            # D = sp.csr_matrix(np.multiply(np.ones((n_samples,n_features)),D))
            D_X =  _add_sparse_column(X,D)
            
            # Then perform the main division
            # ...Find a better way to add a numpy ndarray to a sparse matrix
            np.divide(X.data * (self.k + 1), D_X.data, X.data)
            # np.divide(X.data * (self.k + 1), sp.csr_matrix(np.add(X.todense(), D)).data, X.data)

        elif self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            if not hasattr(self, "_idf_diag"):
                raise ValueError("idf vector not fitted")
            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    @property
    def idf_(self):
        if hasattr(self, "_idf_diag"):
            return np.ravel(self._idf_diag.sum(axis=0))
        else:
            return None
        
def _add_sparse_column(sparse,column):
    addition = sp.lil_matrix(sparse.shape)
    sparse_coo = sparse.tocoo()
    for i,j,v in zip(sparse_coo.row, sparse_coo.col, sparse_coo.data):
        addition[i,j] = v + column[i,0]
    return addition.tocsr()

def _class_frequencies(X, y):
    """Count the number of non-zero values for each class y in sparse X."""

    labels = np.unique(y)
    if len(labels) > 2:
        raise ValueError("Delta works only with binary classification problems")

    # Indices for each type of labels in y
    N1 = np.where(y == labels[0])[0]
    N2 = np.where(y == labels[1])[0]

    # Number of positive documents that each term appears on
    df1 = np.bincount(X[N1].nonzero()[1], minlength=X.shape[1])
    # Number of negative documents that each term appears on
    df2 = np.bincount(X[N2].nonzero()[1], minlength=X.shape[1])

    return N1.shape[0], df1, N2.shape[0], df2


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)
