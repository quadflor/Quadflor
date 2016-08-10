import os
import dill as pickle
import zlib

import numpy as np
from scipy.sparse import csr_matrix, issparse

RELEVANT_OPTIONS = ['terms', 'concepts', 'graph_scoring_method', 'binary', 'hierarchical',
                    'idf', 'norm', 'bm25', 'toy_size', 'synsets']

# TODO path points to directory
class Persister:
    def __init__(self, DATA_PATHS, options):
        """Persister automatically re-saves when the configuration specified in the constant
        RELEVANT_OPTIONS changes, or when the modified-date or file length of the file changes.
        Set options.repersist = True to force re-calculation
        Set options.persist_to to path where to save files.

        Parameters
        ----------
        options: Options object from OptionsParser
            The options from the run method.
        """
        self.options = options
        self.repersist = options.repersist
        self.persist_to = options.persist_to
        self.file_path = DATA_PATHS[options.data_key]['X']

    def _delete_old_files(self):
        if os.path.exists(self.persist_to):
            for fname in os.listdir(self.persist_to):
                if fname.startswith(self._persist_name_head()):
                    os.remove(os.path.join(self.persist_to, fname))

    def persist(self, X, y, thesaurus):
        """
        Save the data and the processed thesaurus.

        Parameters
        ----------
        X: sparse matrix
            The train data: Will be compressed.
        y: sparse matrix
            The label data: Will be compressed.
        thesaurus: ThesaurusReader
            ThesaurusReader object: Will be pickled.
        """
        print('Persisting features to disk')
        self._delete_old_files()
        self._save(self._persist_name('X'), X)
        self._save(self._persist_name('y'), y)
        with open(self._persist_name('TR'), mode='wb') as f:
            pickle.dump(thesaurus, f)

    def read(self):
        """
        Reads from persisted files.

        Returns
        -------
        sparse matrix
            The train data
        sparse matrix
            The label data
        ThesaurusReader
            Unpickled ThesaurusReader object
        """
        print('Reading persisted features')
        X = self._load_sparse_csr(self._persist_name('X'))
        y = self._load_sparse_csr(self._persist_name('y'))
        with open(self._persist_name('TR'), mode='rb') as f:
            tr = pickle.load(f)
        return X, y, tr

    def is_saved(self):
        """
        Returns
        -------
        bool
            Whether or not the current configuration is found in persistence folder.
        """
        return not self.repersist and os.path.exists(self.persist_to) and os.path.exists(
                self._persist_name('X')) and os.path.exists(self._persist_name('y')) and os.path.exists(
                self._persist_name('TR'))

    def _persist_name(self, ending):
        change_date = os.path.getmtime(self.file_path)
        size = os.path.getsize(self.file_path)
        file_name = self._persist_name_head() + str(int(change_date)) + str(size) + ending + '.npz'
        persistence_name = os.path.join(self.persist_to, file_name)
        return persistence_name

    def _persist_name_head(self):
        if self.options.fulltext:
            name = self.file_path.replace(os.sep, '')
        else:
            name = self.file_path.split(os.sep)[-1]
        opt_str = self._relevant_options()
        name_head = name + str(zlib.adler32(opt_str.encode('ascii')))
        return name_head

    def _relevant_options(self):
        filtered_opts = [(k, self.options.__dict__[k]) for k in RELEVANT_OPTIONS]
        return str(sorted(filtered_opts))

    def _save(self, filename, array):
        if not os.path.exists(self.persist_to):
            os.mkdir(self.persist_to)
        if issparse(array):
            np.savez_compressed(filename, data=array.data, indices=array.indices,
                                indptr=array.indptr, shape=array.shape)
        else:
            np.savez_compressed(filename, arr=array)

    @staticmethod
    def _load_sparse_csr(filename):
        loader = np.load(filename)
        try:
            return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                              shape=loader['shape'])
        except KeyError:
            return loader['arr']
