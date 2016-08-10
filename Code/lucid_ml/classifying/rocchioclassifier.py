# Author: Robert Layton <robertlayton@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#
# License: BSD 3 clause

from sklearn.base import BaseEstimator
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestCentroid, nearest_centroid
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from scipy import sparse as sp
from cProfile import label


class RocchioClassifier(NearestCentroid):
    """
    Classifies documents in a Rocchio fashion, but is also capable of multi-label classification. A centroid
    of a class is computed as the average of all training samples that the class is assigned to.
    
    Parameters
    ----------
    metric : string, default='euclidean'
        The distance metric to use for distance measurement. 'euclidean' and 'cosine' are supported.
    shrink_threshold : See documentation of NearestCentroid class in scikit-learn package.
    k : int, default=5
        The number of labels to be assigned to a sample.
    """
    
    def __init__(self, metric = 'euclidean', shrink_threshold = None, k=5):
        NearestCentroid.__init__(self, metric, shrink_threshold)
        self.k = k
        
        
    def fit(self, X, y):
        """
        Fit the RocchioClassifer model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
            Note that centroid shrinking cannot be used with sparse matrices.
        y : array, shape = [n_samples]
            Target values (integers)
        """
        
        is_X_sparse = sp.issparse(X)
        if is_X_sparse and self.shrink_threshold:
            raise ValueError("threshold shrinking not supported"
                             " for sparse input")
        
        n_samples, n_features = X.shape
        
        # take out the labels that don't occur in the gold standard
        self.y = y
        _, cols = y.nonzero()
        label_occurs = np.unique(cols)
        
        # need to memorize what the original index was in the label-matrix
        mem_original_mapping = np.arange(0, y.shape[1])
        self._mem_original_mapping = mem_original_mapping[label_occurs]
        
        y = y[:,label_occurs]
        
        #self.classes_ = classes = range(y.shape[1])
        self.classes_ = []
        self._y = np.empty(y.shape, dtype=np.int)
        for k in range(self._y.shape[1]):
            classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes)

        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')

        # Mask mapping each class to its members.
        self.centroids_ = np.empty((n_classes, n_features), dtype=np.float64)

        # compute the centroids
        for cur_class in range(n_classes):
            center_mask = y[:,cur_class] == 1
            center_mask = center_mask.todense()
            
            if is_X_sparse:
                center_mask = np.where(center_mask)[0]

            self.centroids_[cur_class] = X[center_mask].mean(axis=0)
            


    def predict(self, X):
        """
        Predicts the classes for the samples. Takes the top k classes with smallest distance.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Prediction vector, where n_samples in the number of samples and
            n_features is the number of features.
        """
        predictions = csr_matrix((X.shape[0], self.y.shape[1]), dtype=np.int)
        
        topNIndices, _ = self._get_closest_centroids(X)
        
        for entry, label_list in enumerate(topNIndices):
            for label in label_list:
                predictions[entry, label] = 1
        return predictions

    def predict_proba(self, X):
        """
        Returns a matrix for each of the samples to belong to each of the classes.
        The matrix has shape = [n_samples, n_classes] where n_samples is the
        size of the first dimension of the input matrix X and n_classes is the number of
        classes as determined from the parameter 'y' obtained during training.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Prediction vector, where n_samples in the number of samples and
            n_features is the number of features.
        """
        probabilities = np.zeros((X.shape[0], self.y.shape[1]), dtype=np.float64)
        distances = (pairwise_distances(X, self.centroids_, metric=self.metric))
        
        # in order to get probability like values, we ensure that the closer
        # the distance is to zero, the closer the probability is to 1
        if(self.metric == 'cosine'):
            distances = 1 - distances
        else:
            # in the case of euclidean distance metric we need to normalize by the largest distance
            # to get a value between 0 and 1
            distances = 1 - (distances / distances.max())
        
        # map back onto a matrix containing all labels
        probabilities[:,self._mem_original_mapping] = distances
        
        return probabilities
        
    def _get_closest_centroids(self, X):
        distances = self.predict_proba(X)
        topNIndices = np.apply_along_axis(lambda x: list(np.argsort(x)[-self.k:]), 1, distances)
        topNDistances = np.apply_along_axis(lambda x: list(np.sort(x)[-self.k:]), 1, distances)
        return [topNIndices, topNDistances]
