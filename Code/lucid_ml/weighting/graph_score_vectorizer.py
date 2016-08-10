from collections import defaultdict

import networkx as nx
import scipy.sparse as sp

from utils.nltk_normalization import NltkNormalizer


# noinspection PyStatementEffect
class GraphVectorizer:
    """
    Use graph activation to extract feature vector.

    Parameters
    ----------
    method: str, default='degree'
        one of ['degree', 'betweenness', 'pagerank', 'hits', 'closeness', 'katz']
    analyzer: Callable, default=NltkNormalizer().split_and_normalize
        Analyzer method. Insert ConceptAnalyzer.analyze here if you want to use concepts. Otherwise
        normal terms are used.
    """

    def __init__(self, method='degree', analyzer=NltkNormalizer().split_and_normalize):
        self.analyze = analyzer
        self.method = method
        self.methods_on_digraph = {'hits', 'pagerank', 'katz'}
        self._get_scores = {'degree': nx.degree, 'betweenness': nx.betweenness_centrality,
                            'pagerank': nx.pagerank_scipy, 'hits': self._hits, 'closeness': nx.closeness_centrality,
                            'katz': nx.katz_centrality}[method]
        # Add a new value when a new vocabulary item is seen
        self.vocabulary = defaultdict()
        self.vocabulary.default_factory = self.vocabulary.__len__

    def fit(self, docs):
        """
        Use analyzer on docs to learn vocabulary of tokens.

        Parameters
        ----------
        docs: collections.Iterable[str]
            The raw documents.
        """
        for doc in docs:
            for w in self.analyze(doc):
                self.vocabulary[w]

    def transform(self, docs):
        """
        Apply graph activation and return feature vector

        Parameters
        ----------
        docs: collections.Iterable[str]
            The raw documents

        Returns
        -------
        scipy.sparse.csr_matrix
            The feature vector

        """
        scores = []
        for doc in docs:
            graph = nx.DiGraph() if self.method in self.methods_on_digraph else nx.Graph()
            split_doc = self.analyze(doc)
            for node_a, node_b in zip(split_doc[:len(split_doc) - 1], split_doc[1:]):
                graph.add_edge(node_a, node_b)
            scores.append(self._get_scores(graph))
        return self._make_sparse(scores)

    def fit_transform(self, raw_docs, y=None):
        """
        Learn a vocabulary of tokens using analyzer and apply graph activation to return feature vector.
        Slightly more efficient than first fit than transform.

        Parameters
        ----------
        raw_docs: collections.Iterable[str]
            The raw documents
        y:
            ignored

        Returns
        -------
        scipy.sparse.csr_matrix
            The feature vector

        """
        scores = []
        for doc in raw_docs:
            graph = nx.DiGraph() if self.method in self.methods_on_digraph else nx.Graph()
            split_doc = self.analyze(doc)
            if split_doc:
                # Add first feature to self.vocabulary
                self.vocabulary[split_doc[0]]
                for node_a, node_b in zip(split_doc[:len(split_doc) - 1], split_doc[1:]):
                    self.vocabulary[node_b]
                    graph.add_edge(node_a, node_b)
                scores.append(self._get_scores(graph))
            else:  # No feature found or only stopwords, etc.
                scores.append({})
        return self._make_sparse(scores)

    @staticmethod
    def _hits(graph):
        authority, hubness = nx.hits_scipy(graph, tol=1e-1, max_iter=100)
        return {a[0]: a[1] + hubness[a[0]] for a in authority.items()}

    def _make_sparse(self, scores):
        n_features = len(self.vocabulary)
        result = sp.csr_matrix((0, n_features))
        for score in scores:
            sparse_score = sp.dok_matrix((1, n_features))
            for s in score.items():
                sparse_score[0, self.vocabulary[s[0]]] = s[1]
            result = sp.vstack((result, sp.csr_matrix(sparse_score)))
        return result
