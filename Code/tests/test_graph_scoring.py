# from timeit import default_timer
from unittest import TestCase

import numpy as np

# from utils.Extractor import load_dataset
from weighting.graph_score_vectorizer import GraphVectorizer
from weighting.statistical.concept_analysis import ConceptAnalyzer


class TestGraphScoring(TestCase):
    def test_fit_transform_simple_degree(self):
        doc = ['cats eat dogs eat people eat']
        scorer = GraphVectorizer()
        res = scorer.fit_transform(doc).todense()
        np.testing.assert_array_almost_equal([[1, 3, 1, 1]], res)

    def test_first_fit_then_transform_simple_degree(self):
        docs = ['cats eat dogs eat people eat', 'bears eat lots']
        scorer = GraphVectorizer()
        scorer.fit(docs)
        res = scorer.transform(docs)
        np.testing.assert_array_almost_equal([[1, 3, 1, 1, 0, 0], [0, 2, 0, 0, 1, 1]], res.todense())

    def test_other_method(self):
        doc = ['cats eat dogs eat people eat']
        scorer = GraphVectorizer(method='betweenness')
        res = scorer.fit_transform(doc).todense()
        np.testing.assert_array_almost_equal([[0, 1, 0, 0]], res)

    def test_that_other_methods_run_without_error(self):
        doc = ['cats eat dogs eat people eat']

        def try_method(method):
            scorer = GraphVectorizer(method=method)
            return scorer.fit_transform(doc)

        methods = ['degree', 'betweenness', 'pagerank', 'hits', 'closeness', 'katz']
        for method in methods:
            res = try_method(method)
            # print(method)
            # print(res)

    def test_analyzer(self):
        docs = ['ipsum dolor sit amet']
        thesaurus = {'00': {'prefLabel': ['ipsum'], 'broader': [],
                            'narrower': ['01', '02'], 'altLabel': []},
                     '01': {'prefLabel': ['dolor'], 'broader': ['00'],
                            'narrower': [], 'altLabel': ['amet']},
                     '02': {'prefLabel': ['sit'], 'broader': ['00'],
                            'narrower': [], 'altLabel': []},
                     }
        analyzer = ConceptAnalyzer(thesaurus).analyze
        scorer = GraphVectorizer(analyzer=analyzer)
        res = scorer.fit_transform(docs).todense()
        np.testing.assert_array_almost_equal([[1, 2, 1]], res)

    def test_analyzer_no_concepts(self):
        docs = ['no concepts here']
        thesaurus = {'00': {'prefLabel': ['ipsum'], 'broader': [],
                            'narrower': ['01', '02'], 'altLabel': []},
                     '01': {'prefLabel': ['dolor'], 'broader': ['00'],
                            'narrower': [], 'altLabel': ['amet']},
                     '02': {'prefLabel': ['sit'], 'broader': ['00'],
                            'narrower': [], 'altLabel': []},
                     }
        analyzer = ConceptAnalyzer(thesaurus).analyze
        scorer = GraphVectorizer(analyzer=analyzer)
        res = scorer.fit_transform(docs).todense()
        np.testing.assert_array_almost_equal([[]], res)

    def test_standard_lemmatizer(self):
        doc = ['cats dog cat dogs']
        scorer = GraphVectorizer()
        res = scorer.fit_transform(doc).todense()
        np.testing.assert_array_almost_equal([[1, 1]], res)

        # def test_speed(self):
        #     data, _, _ = load_dataset()
        #     start = default_timer()
        #     scorer = GraphVectorizer()
        #     scorer.fit_transform(data[:15000])
        #     print(default_timer() - start)
