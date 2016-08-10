import os
import shutil
import tempfile
from unittest import TestCase

import numpy as np
from scipy.sparse import csr_matrix

import run
from tests.test_thesaurus_reader import make_thes, setify
from utils import Extractor
from utils.persister import Persister
from utils.thesaurus_reader import ThesaurusReader


class TestPersister(TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp()

        def options():
            pass

        parser = run._generate_parser()
        default_values = parser.get_default_values().__dict__
        for default in default_values.items():
            setattr(options, default[0], default[1])
        options.thesaurus = 'th'
        options.persist_to = self.path
        options.goldstd = 'g'
        options.concepts = True
        options.one_fold = True
        self.options = options
        Extractor.DATA_PATHS[options.data_key] = tempfile.NamedTemporaryFile(delete=False, dir=self.path).name

    def tearDown(self):
        shutil.rmtree(self.path)

    def test_persist_and_read(self):
        persister = Persister(self.options)
        tr = ThesaurusReader(make_thes())
        X = csr_matrix([[1]])
        y = csr_matrix([[2]])
        persister.persist(X, y, tr)
        self.assertTrue(persister.is_saved())
        np.testing.assert_array_equal(X.todense(), persister.read()[0].todense())
        np.testing.assert_array_equal(y.todense(), persister.read()[1].todense())

        expected = {
            'root': {'narrower': ['topc'], 'broader': [], 'prefLabel': ['r00t'], 'altLabel': []},
            'topc': {'narrower': ['636688', '636686'], 'broader': ['root'], 'prefLabel': ['reduction'],
                     'altLabel': ['alternative reduction']},
            '636688': {'narrower': [], 'broader': ['topc'], 'prefLabel': ['reduction emission'],
                       'altLabel': []},
            '636686': {'narrower': [], 'broader': ['topc'], 'prefLabel': ['security risk'], 'altLabel': []}
        }
        tr_read = persister.read()[2]

        thesaurus = setify(tr_read.thesaurus)
        expected = setify(expected)
        self.assertEquals(thesaurus, expected)

    def test_persist_and_read_folder(self):
        tempdir = tempfile.mkdtemp()
        Extractor.DATA_PATHS[self.options.data_key] = (tempdir,)
        self.options.fulltext = True
        persister = Persister(self.options)
        persister.persist(csr_matrix([[1]]), csr_matrix([[2]]), ThesaurusReader(make_thes()))
        expected_name = tempdir.replace(os.sep, '')
        onlyfiles = [f for f in os.listdir(tempdir) if os.path.isfile(os.path.join(tempdir, f))]
        for f in onlyfiles:
            self.assertTrue(f.startswith(expected_name))

    def test_persist_and_read_dense_array(self):
        persister = Persister(self.options)
        X = np.array([[1]])
        y = np.array([[2]])
        tr = ThesaurusReader(make_thes())
        persister.persist(X, y, tr)
        self.assertTrue(persister.is_saved())
        np.testing.assert_array_equal(X, persister.read()[0])
        np.testing.assert_array_equal(y, persister.read()[1])
