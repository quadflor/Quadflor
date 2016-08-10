# from timeit import default_timer
from unittest import TestCase

import networkx as nx
# import numpy as np
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.scorer import _PredictScorer

# from utils.Extractor import load_dataset

from utils.metrics import hierarchical_f_measure, hierarchical_f_measure_scorer


class TestHierarchicalFMeasure(TestCase):
    def test_simple_f_score(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([(0, 1), (1, 2), (1, 3)])
        y_true = sp.csr_matrix([[0, 0, 1, 0]])
        y_pred = sp.csr_matrix([[0, 0, 0, 1]])
        self.assertEquals(0.5, hierarchical_f_measure(self.imitate_tr(graph, 0), y_true, y_pred))

    def test_simple_f_score_dense(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([0,1, 2, 3])
        graph.add_edges_from([(0, 1), (1, 2), (1, 3)])
        y_true = np.matrix([[0, 0,1, 0]])
        y_pred = np.matrix([[0, 0,0, 1]])
        self.assertEquals(0.5, hierarchical_f_measure(self.imitate_tr(graph, 0), y_true, y_pred))

    def test_ill_defined(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edges_from([(0, 1), (0, 2)])
        y_true = sp.csr_matrix([[0, 1, 0]])
        y_pred = sp.csr_matrix([[0, 0, 0]])
        self.assertEquals(0, hierarchical_f_measure(self.imitate_tr(graph, 0), y_true, y_pred))

    def test_more_complex(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2, 3, 4, 5])
        graph.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (1, 3)])
        y_true = sp.csr_matrix([[0,0, 0, 0, 1, 1], [0,1, 0, 0, 0, 0]])
        y_pred = sp.csr_matrix([[0,0, 0, 1, 0, 1], [0,0, 0, 0, 0, 1]])
        # int = 4, ta = 5, pa = 4, p = 4/4, r = 4/5, f = 2*(4/4)*(4/5)/(4/4+4/5)
        self.assertEquals((2 * (4 / 5) / (1 + 4 / 5) + 2 * (1 / 1) * (1 / 4) / (1 + 1 / 4)) / 2,
                          hierarchical_f_measure(self.imitate_tr(graph, 0), y_true, y_pred))

    def test_make_scorer(self):
        graph = nx.DiGraph()
        scorer = hierarchical_f_measure_scorer(graph)
        self.assertTrue(callable(scorer))
        self.assertIsInstance(scorer, _PredictScorer)

    def imitate_tr(self, graph, root):
        def tr():
            pass

        tr.nx_graph = graph
        tr.nx_root = root
        return tr
        #
        # def test_speed(self):
        #     _, _, tr = load_dataset('econ62k')
        #     graph = tr.nx_graph
        #
        #     def random_labels():
        #         def set_random_ones(n_nodes):
        #             ids = np.random.choice(n_nodes, 5)
        #             zeros = sp.dok_matrix((1, n_nodes), dtype=np.bool_)
        #             for index in ids:
        #                 zeros[0, index] = True
        #             return zeros
        #
        #         number_of_nodes = graph.number_of_nodes()
        #         matrix = set_random_ones(number_of_nodes)
        #         for i in range(0, 62000):
        #             zeros = set_random_ones(number_of_nodes)
        #             matrix = sp.vstack((matrix, zeros))
        #         return sp.csr_matrix(matrix)
        #
        #     y_true = random_labels()
        #     y_pred = random_labels()
        #     print('random constructed')
        #
        #     start = default_timer()
        #     hierarchical_f_measure(graph, y_true, y_pred)
        #     print(default_timer() - start)
