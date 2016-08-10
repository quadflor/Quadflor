# from timeit import default_timer
from unittest import TestCase

# import numpy as np

from weighting.synset_analysis import SynsetAnalyzer


class TestSynsetAnalyzer(TestCase):
    def test_simple(self):
        doc = 'Gender and the internet.'
        a = SynsetAnalyzer()
        res = a.analyze(doc)
        expected = ['sex.n.04', 'internet.n.01']
        self.assertEquals(expected, res)

    def test_lemmatize_needed(self):
        doc = 'I am loving you.'
        a = SynsetAnalyzer()
        res = a.analyze(doc)
        expected = ['love.v.03']
        self.assertEquals(expected, res)

    # def test_speed(self):
    #     doc = 'Buyer power and product innovation : empirical evidence from the German food sector'
    #     times = []
    #     a = SynsetAnalyzer()
    #     for i in range(0, 500):
    #         start = default_timer()
    #         a.analyze(doc)
    #         times.append(default_timer()-start)
    #     print(np.mean(times) * 62000)
