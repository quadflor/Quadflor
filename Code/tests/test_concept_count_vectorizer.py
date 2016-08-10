from timeit import default_timer
from tempfile import NamedTemporaryFile, mkdtemp
from unittest import TestCase

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import run
from lucid_ml.weighting.concept_analysis import ConceptAnalyzer

from utils.Extractor import load_dataset
from utils.persister import Persister


class TestConceptCountVectorizer(TestCase):
    def test_simple_transform(self):
        # alphabetic: adipisici amet consectetur dolor elit ipsum lorem sit
        doc = 'Lorem Lorem ipsum dolor sit amet'
        thesaurus = {'00': {'prefLabel': ['ipsum'], 'broader': ['0b'], 'related': ['0r'],
                            'narrower': ['0n'], 'altLabel': []},
                     '01': {'prefLabel': ['dolor'], 'broader': ['1b'], 'related': ['1r'],
                            'narrower': ['1n'], 'altLabel': ['amet']},
                     }
        cf = ConceptAnalyzer(thesaurus)
        expected = ['00', '01', '01']
        self.assertEquals(expected, cf.analyze(doc))

    def test_vocabulary(self):
        docs = ['Lorem ipsum', 'Lorem Lorem ipsum Dolor sit AMET', 'consectetur adipisici elit']
        thesaurus = {'00': {'prefLabel': ['ipsum'], 'broader': ['0b'], 'related': ['0r'],
                            'narrower': ['0n'], 'altLabel': []},
                     '01': {'prefLabel': ['dolor'], 'broader': ['1b'], 'related': ['1r'],
                            'narrower': ['1n'], 'altLabel': ['amet']},
                     }
        vocabulary = {'00': 1, '01': 0}
        cf = ConceptAnalyzer(thesaurus)
        counter = CountVectorizer(analyzer=cf.analyze, vocabulary=vocabulary)
        res = counter.fit_transform(docs).todense()
        np.testing.assert_array_almost_equal(res, [[0, 1], [2, 1], [0, 0]])

    def test_vocabulary_with_entity_ids(self):
        docs = ['Lorem ipsum', 'Lorem Lorem ipsum Dolor sit AMET', 'consectetur adipisici elit']
        thesaurus = {'13542-1': {'prefLabel': ['ipsum'], 'broader': ['0b'], 'related': ['0r'],
                                 'narrower': ['0n'], 'altLabel': []},
                     '13542-4': {'prefLabel': ['dolor'], 'broader': ['1b'], 'related': ['1r'],
                                 'narrower': ['1n'], 'altLabel': ['amet']},
                     }
        vocabulary = {'13542-1': 1, '13542-4': 0}
        cf = ConceptAnalyzer(thesaurus)
        counter = CountVectorizer(analyzer=cf.analyze, vocabulary=vocabulary)
        res = counter.fit_transform(docs).todense()
        np.testing.assert_array_almost_equal(res, [[0, 1], [2, 1], [0, 0]])

    def test_no_alt_label(self):
        docs = ['Lorem ipsum', 'Lorem Lorem ipsum Dolor sit DOLOR', 'consectetur adipisici elit']
        thesaurus = {'13542-1': {'prefLabel': ['ipsum'], 'broader': ['0b'], 'related': ['0r'],
                                 'narrower': ['0n']},
                     '13542-4': {'prefLabel': ['dolor'], 'broader': ['1b'], 'related': ['1r'],
                                 'narrower': ['1n']},
                     }
        vocabulary = {'13542-1': 1, '13542-4': 0}
        cf = ConceptAnalyzer(thesaurus)
        counter = CountVectorizer(analyzer=cf.analyze, vocabulary=vocabulary)
        res = counter.fit_transform(docs).todense()
        np.testing.assert_array_almost_equal(res, [[0, 1], [2, 1], [0, 0]])

    def test_read_files(self):
        docs = ['Lorem ipsum', 'Lorem Lorem ipsum Dolor sit AMET', 'consectetur adipisici elit']
        thesaurus = {'13542-1': {'prefLabel': ['ipsum'], 'broader': ['0b'], 'related': ['0r'],
                                 'narrower': ['0n'], 'altLabel': []},
                     '13542-4': {'prefLabel': ['dolor'], 'broader': ['1b'], 'related': ['1r'],
                                 'narrower': ['1n'], 'altLabel': ['amet']},
                     }
        vocabulary = {'13542-1': 1, '13542-4': 0}
        fnames = []
        for doc in docs:
            file = NamedTemporaryFile(mode='w', delete=False)
            fnames.append(file.name)
            print(doc, file=file)
        cf = ConceptAnalyzer(thesaurus, input='filename')
        counter = CountVectorizer(analyzer=cf.analyze, vocabulary=vocabulary, input='filename')
        res = counter.fit_transform(fnames).todense()
        np.testing.assert_array_almost_equal(res, [[0, 1], [2, 1], [0, 0]])

    def test_persist(self):
        # alphabetic: adipisici amet consectetur dolor elit ipsum lorem sit
        docs = ['Lorem ipsum', 'Lorem Lorem ipsum Dolor sit AMET', 'consectetur adipisici elit']
        thesaurus = {'13542-1': {'prefLabel': ['ipsum'], 'broader': ['0b'], 'related': ['0r'],
                                 'narrower': ['0n'], 'altLabel': []},
                     '13542-4': {'prefLabel': ['dolor'], 'broader': ['1b'], 'related': ['1r'],
                                 'narrower': ['1n'], 'altLabel': ['amet']},
                     }
        vocabulary = {'13542-1': 1, '13542-4': 0}
        tempdir_data = mkdtemp()
        fnames = []
        for doc in docs:
            file = NamedTemporaryFile(mode='w', delete=False, dir=tempdir_data)
            fnames.append(file.name)
            print(doc, file=file)
        tempdir = mkdtemp()
        cf = ConceptAnalyzer(thesaurus, persist=True, persist_dir=tempdir, input='filename', file_path=tempdir_data)
        for fname in fnames:
            cf.analyze(fname)
        cf.persistence_file.close()
        cf2 = ConceptAnalyzer(thesaurus, persist=True, persist_dir=tempdir, input='filename', file_path=tempdir_data)
        for fname in fnames:
            with open(fname, mode='w') as file:
                print('bullshit', file=file)
        counter = CountVectorizer(analyzer=cf2.analyze, vocabulary=vocabulary)
        res = counter.fit_transform(fnames).todense()
        print(res)
        np.testing.assert_array_almost_equal(res, [[0, 1], [2, 1], [0, 0]])

    def test_speed(self):
        doc = self.text() * 200
        tempdir_data = mkdtemp()
        file = NamedTemporaryFile(mode='w', delete=False, dir=tempdir_data)
        print(doc, file=file)
        _, _, tr = load_dataset({'econ62k': {
            "X": "../../../Resources/Goldstandard/formatted_econbiz-annotation-62k-titles.csv",
            "y": "../../../Resources/Goldstandard/econbiz-stw9-formatted.csv",
            "thes": "../../../Resources/Ontologies/stw.json"
        }})
        cf = ConceptAnalyzer(tr.thesaurus, persist=True, persist_dir=mkdtemp(), input='filename',
                             file_path=tempdir_data)
        times = []
        for _ in range(10):
            start = default_timer()
            cf.analyze(file.name)
            times.append(default_timer() - start)
        print(np.round(np.mean(times), decimals=2))

    def text(self):
        return """
        Most non-elderly Americans get their health insurance through
        either their own employment, or the employment of family members.
        Thus, evidence that rates of private health insurance coverage have
        fallen over time have caused great concern. For example, Table 1
        (from Farber and Levy (1998) Table 13) shows that the fraction of
        private sector workers aged 20 to 65 who were covered by their own
        employer's insurance fell from 72 to 65% between 1979 and 1997.
        The decline was much more dramatic among workers without a high
        school education; among these workers coverage fell from 67 to 50%.
        A closer inspection of Table 1 suggests however, that the
        decline in private health insurance coverage slowed to a halt
        between 1993 and 1997. This paper provides additional confirmation
        of this finding. Using data from three different sources, we find
        that in contrast to the preceding two decades, there has been
        little overall decline in private health insurance coverage in the
        1990s. This finding holds even for less-educated single mothers,
        a group of particular concern to policy makers in this era of
        welfare reform.
        The paper begins with some theoretical considerations
        regarding the reasons why health insurance is provided by
        employers. We continue with an overview of the available data for
        the period 1987 to 1997, and with a discussion of trends in health
        insurance coverage over that period. Finally, we offer some
        observations about three hypotheses which may be used to explain
        the earlier decline in health insurance coverage, as well as the
"""
