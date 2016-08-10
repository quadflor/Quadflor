#!/usr/bin/env/python3
# -*- coding:utf-8 -*-
import networkx as nx
from collections import defaultdict, deque
from math import log

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from math import log


class SpreadingActivationTransformer(BaseEstimator, TransformerMixin):
    '''
    Create a SpreadingActivation object
    parameters:
    hierarchy -- the hierarchy of concepts as a network x graph
    root -- the root node of the hierarchy
    method -- activation method: one of 'basic', 'bell', 'bellog', 'children'
    decay -- decay factor used by the 'basic' activation method
    vocabulary (optional) -- mapping from hierarchy nodes to matrix indices
    feature_names (optional) -- mapping from matrix indices to hierarchy nodes
    '''
    def __init__(self, hierarchy, root, method='basic', decay=1.0, vocabulary=None, feature_names=None):
        self.method = method.lower()
        if self.method not in ["basic", "bell", "belllog", "children", "binary"]:
            raise ValueError
        self.hierarchy = hierarchy
        self.root = root

        # if thesaurus does not use matrix indices as nodes,
        # we need some vocabulary and feature_names mappings
        self.vocabulary = vocabulary
        self.feature_names = feature_names

        # decay is used for basic activation method
        self.decay = decay

        
    def _score(self, freq, scores, row, col, memoization=None):
        mem = memoization if memoization is not None else [False] * scores.shape[1]

        # memoization hit
        if mem[col]: return scores[row, col]
        
        children = self.hierarchy.successors(self.feature_names[col] if self.feature_names else col)
        if len(children) == 0:
            # Base case for leaves
            scores[row, col] = freq[row, col]
            mem[col] = True
            return scores[row, col]

        # recursively compute children score
        score = float(0)
        for child in children:
            child_idx = self.vocabulary[child] if self.vocabulary else child
            score += self._score(freq, scores, row, child_idx, memoization=mem)

        # scale them with some method specific factor
        if self.method in ["bell", "belllog"]:
            k = nx.shortest_path_length(self.hierarchy, self.root, self.feature_names[col] if self.feature_names else col)
            print(k+1, self.levels[k+1])
            print("Count of children:", len(children))
            denom = self.levels[k+1]
            if self.method == "belllog": denom = log(denom, 10) #TODO problem when zero
            score *= 1.0 / denom
        elif self.method == "children":
            score *= 1.0 / len(children)
        elif self.method == "basic":
            score *= self.decay 

        # add the freq of the concept just now since it should not be scaled
        score += freq[row, col]


        scores[row, col] = score
        mem[col] = True

        return scores[row, col]

    def partial_fit(self, X, y=None):
        return self

    def fit(self, X, y=None):
        # the bell methods require additional information
        if self.method in ["bell", "belllog"]:
            # precompute node count by level
            self.levels = defaultdict(int)
            for node in self.hierarchy.nodes():
                l = nx.shortest_path_length(self.hierarchy, self.root, node)
                self.levels[l] += 1


            print(self.levels)
        return self
               
    def transform(self, X, y=None):
        n_records, n_features = X.shape
        # lil matrix can be modified efficiently
        # especially when row indices are sorted
        scores = sp.lil_matrix((n_records, n_features), dtype=np.float32)
        for row in range(n_records):
            self._score(X, scores, row, self.root)
        return sp.csr_matrix(scores)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


def write_dotfile(path, data, shape):
    def identifier(record, node):
        return str(record) + '.' + str(node)
    nx, ny = shape
    with open(path, 'w') as f:
        print("digraph G {", file=f)
        print("node [shape=rect]", file=f)
        for record in range(nx):
            for feature in range(ny):
                s = identifier(record, feature)
                s += " [label=\""
                for key, value in data.items():
                    s += key + ":\t%.2f"%value[record,feature] + "\\n"
                s += "\"]"
                print(s, file=f)

            for edge in toy.edges():
                src, dst = edge
                print(identifier(record, src), "->", identifier(record, dst), file=f)
        print("}", file=f)

if __name__ == "__main__":
    import random
    # toy hierarchy
    toy = nx.DiGraph()
    toy.add_nodes_from([0,1,2,3,4,5,6,7,8,9,10,11,12])
    toy.add_edges_from([(0,1), (0,2), (0,3), (1,4), (1, 5), (2,6), (2,7), (2,8), (2,9), (2,10),
        (3,7),(4,11),(4,12)])

    # toy shape
    n_records = 3
    n_features = len(toy.nodes())

    # fill with random values
    freq = np.ndarray(shape=(n_records, n_features), dtype=np.int8)
    for i in range(n_records):
        for j in range(n_features):
            freq[i,j] = random.randint(0,4)

    freq = sp.csr_matrix(freq)

    print("Initial frequency values as CSR matrix")
    print("=" * 42)
    print(freq)
    print("=" * 42)

    # initialize methods
    basic = SpreadingActivationTransformer(toy, 0, method="basic")
    bell = SpreadingActivationTransformer(toy, 0, method="bell")
    belllog = SpreadingActivationTransformer(toy, 0, method="belllog")
    children = SpreadingActivationTransformer(toy, 0, method="children")

    # apply them
    basic_scores = basic.fit_transform(freq)
    children_scores = children.fit_transform(freq)
    bell_scores = bell.fit_transform(freq)
    belllog_scores = belllog.fit_transform(freq)

    print("Computed values as CSR matrix (with children spreading activation)")
    print("=" * 42)
    print(children_scores)
    print("=" * 42)

    # put them in a dict
    data_dict = { 
            "freq" : freq,
            "basic" : basic_scores,
            "children" : children_scores,
            "bell" : bell_scores,
            "bellog" : bell_scores }

    # for some pretty output
    write_dotfile("more_toys.dot", data_dict, shape=freq.shape)

class InverseSpreadingActivation(BaseEstimator, TransformerMixin):
    def __init__(self, hierarchy, multilabelbinarizer, decay=0.4, firing_threshold=1.0, verbose=0, use_weights=True):
        self.hierarchy = hierarchy
        self.decay = decay
        self.firing_threshold = firing_threshold
        self.use_weights = use_weights
        self.verbose = verbose
        self.mlb = multilabelbinarizer

    def fit(self, X, Y):
        n_samples = X.shape[0]
        F = self.firing_threshold
        decay = self.decay
        coef_ = np.zeros(shape=(X.shape[1]), dtype=np.float64)
        fired_ = np.zeros(shape=(X.shape[1]), dtype=np.bool_)
        _, I, V = sp.find(Y)
        coef_[I] += np.divide(V[I], X.shape[0])

        markers = deque(I)
        while markers:
            i = markers.popleft()
            if coef_[i] >= F and not fired[i]:
                #fire
                for j in self.hierarchy.neighbors(i):
                    if self.use_weights:
                        coef_[j] += coef[i] * decay * hierarchy[i][j]['weight']
                    else:
                        coef_[j] += coef[i] * decay 
                    if coef_[j] >= F:
                        coef_[j] = F
                        markers.append(n)

        self.coef_ = coef_
        return self

    def transform(self, X):
        Xt = X + X * self.coef_
        return Xt

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X)


def bell_reweighting(tree, root, sublinear=False):
    # convert the hierarchy to a tree if make_bfs_tree is true

    distance_by_target = nx.shortest_path_length(tree, source=root)

    level_count = defaultdict(int)
    for val in distance_by_target.values():
        level_count[val] += 1

    for edge in tree.edges():
        parent, child = edge
        if sublinear:
            # use smoothed logarithm
            tree[parent][child]['weight'] = 1.0 / log(1 + level_count[distance_by_target[child]], 10)
        else:
            tree[parent][child]['weight'] = 1.0 / level_count[distance_by_target[child]]

    return tree

def children_reweighting(tree):
    for node in tree.nodes():
        children = tree.successors(node)
        n_children = len(children)
        for child in children:
            tree[node][child]['weight'] = 1.0 / n_children

    return tree

class SpreadingActivation(BaseEstimator, TransformerMixin):
    '''
    weighting == None implies equal weights to all edges
    weighting == bell, belllog requires root to be defined and assert_tree should be true
    '''
    def __init__(self, hierarchy, decay=1, firing_threshold=0, verbose=10, weighting=None, root=None, strict=False):
        self.hierarchy = hierarchy
        self.decay = decay
        self.firing_threshold = firing_threshold
        self.verbose = verbose 
        self.strict = strict
        self.root = root
        self.weighting = weighting.lower() if weighting is not None else None
        assert self.weighting in [None, "bell", "belllog", "children", "basic"]

    def fit(self, X, y=None):
        if self.weighting == "bell":
            assert self.root is not None
            self.hierarchy = bell_reweighting(self.hierarchy, self.root, sublinear=False)
        elif self.weighting == "belllog":
            assert self.root is not None
            self.hierarchy = bell_reweighting(self.hierarchy, self.root, sublinear=True)
        elif self.weighting == "children":
            self.hierarchy = children_reweighting(self.hierarchy)
        return self

    def transform(self, X):
        F = self.firing_threshold
        hierarchy = self.hierarchy
        decay = self.decay
        if self.verbose: print("[SA] %.4f concepts per sample."%(float(X.getnnz()) / X.shape[0]))
        if self.verbose: print("[SA] Starting Spreading Activation")
        X_out = sp.lil_matrix(X.shape,dtype=X.dtype)
        fired = sp.lil_matrix(X.shape,dtype=np.bool_)
        I, J, V = sp.find(X)
        X_out[I,J] = V
        markers = deque(zip(I,J))
        while markers:
            i, j = markers.popleft()
            if X_out[i,j] >= F and not fired[i,j]:
                #markers.extend(self._fire(X_out, i, j))
                fired[i,j] = True 
                for target in hierarchy.predecessors(j):
                    if self.weighting:
                        X_out[i,target] += X_out[i,j] * decay * hierarchy[target][j]['weight']     
                    else:
                        X_out[i,target] += X_out[i,j] * decay 

                    if X_out[i, target] >= F:
                        if self.strict: A[i,target] = F
                        markers.append((i,target))

        if self.verbose: print("[SA] %.4f fired per sample."%(float(fired.getnnz()) / X.shape[0]))
        return sp.csr_matrix(X_out)


    def _fire(self, A, i, j):
        F = self.firing_threshold
        hierarchy = self.hierarchy
        decay = self.decay
        markers = deque()
        for target in hierarchy.predecessors(j):
            if self.weighting:
                A[i,target] += A[i,j] * decay * hierarchy[target][j]['weight']     
            else:
                A[i,target] += A[i,j] * decay 

            if A[i, target] >= F:
                if self.strict: A[i,target] = F
                markers.append((i, target))
        return markers

class OneHopActivation(BaseEstimator, TransformerMixin):
    def __init__(self, hierarchy, decay=0.4, child_treshold=2,verbose=0):
        self.hierarchy = hierarchy
        self.decay = decay
        self.child_threshold = child_treshold
        self.verbose = verbose


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        hierarchy = self.hierarchy
        decay = self.decay
        threshold = self.child_threshold
        verbose = self.verbose

        n_hops = 0
        if verbose: print("[OneHopActivation]")
        X_out = sp.lil_matrix(X.shape, dtype=X.dtype)
        I, J, _ = sp.find(X)
        for i, j in zip(I,J):
            n_children = 0
            sum_children = 0
            for child in hierarchy.successors(j):
                if X[i, child] > 0: # same row i
                    n_children += 1
                    sum_children += X[i, child]
            if n_children >= threshold:
                if verbose: print("Hop", end=" ")
                n_hops += 1
                X_out[i,j] = X[i,j] + sum_children * decay
            else:
                X_out[i,j] = X[i,j]

        if verbose: print("\n[OneHopActivation] %d hops." % n_hops)

        return sp.csr_matrix(X_out)


class BinarySA(BaseEstimator, TransformerMixin):
    ''' Binary Spreading Activation Transformer
        + works in place and on sparse data
    '''
    def __init__(self, hierarchy, assert_tree=False, root=None):
        self.hierarchy = hierarchy
        self.assert_tree = assert_tree
        self.root = root
        
    def fit(self, X, y=None):
        if self.assert_tree:
                assert self.root is not None
                self.hierarchy = nx.bfs_tree(self.hierarchy, self.root)
        return self

    def transform(self, X, y=None):
        ''' From each value in the feature matrix,
        traverse upwards in the hierarchy (including multiple parents in DAGs),
        and set all nodes to one'''
        hierarchy = self.hierarchy
        X_out = np.zeros(X.shape, dtype=np.bool_)
        samples, relevant_topics, _ = sp.find(X)
        for sample, topic in zip(samples, relevant_topics):
            X_out[sample, topic] = 1
            ancestors = nx.ancestors(hierarchy, topic)
            for ancestor in ancestors:
                X_out[sample, ancestor] = 1

        return X_out

                
