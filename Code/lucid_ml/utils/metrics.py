from functools import partial
from warnings import warn

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.sputils import isdense
from sklearn.metrics import make_scorer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils.sparsefuncs import count_nonzero


def hierarchical_f_measure(tr, y_true, y_pred):
    """
    Calculate hierarchical f-measure. This is defined as the f-measure precision and recall
    calculated with the union of the ancestors
    of the given labels (including the labels themselves and excluding the root).

    Parameters
    ----------
    tr: ThesaursReader
        The thesaurus.
    y_true: {sparse matrix, array-like}
        The true labels
    y_pred: {sparse matrix, array-like}
        The predicited labels

    Returns
    -------
    float
        The hierarchical f_measure
    """
    graph = tr.nx_graph
    root = tr.nx_root
    if not sp.issparse(y_true):
        y_true = sp.coo_matrix(y_true)
        y_pred = sp.coo_matrix(y_pred)
    label_scores = []
    for i in range(0, y_true.shape[0]):
        row_true = y_true.getrow(i)
        row_pred = y_pred.getrow(i)
        true_ancestors = set.union(set(row_true.indices), *[nx.ancestors(graph, index) for index in row_true.indices])
        true_ancestors.discard(root)
        pred_ancestors = set.union(set(row_pred.indices), *[nx.ancestors(graph, index) for index in row_pred.indices])
        pred_ancestors.discard(root)
        intersection = len(pred_ancestors & true_ancestors)
        try:
            p = intersection / len(pred_ancestors)
            r = intersection / len(true_ancestors)
            label_scores.append(2 * p * r / (p + r))
        except ZeroDivisionError:
            warn('F_score is ill-defined and being set to 0.0 on samples with no predicted labels',
                 UndefinedMetricWarning, stacklevel=2)
            label_scores.append(0)
    return np.mean(label_scores)


def hierarchical_f_measure_scorer(graph):
    measure = partial(hierarchical_f_measure, graph)
    return make_scorer(measure)


def f1_per_sample(y_true, y_pred):
    if isdense(y_true) or isdense(y_pred):
        y_true = sp.csr_matrix(y_true)
        y_pred = sp.csr_matrix(y_pred)
    sum_axis = 1
    true_and_pred = y_true.multiply(y_pred)
    tp_sum = count_nonzero(true_and_pred, axis=sum_axis)
    pred_sum = count_nonzero(y_pred, axis=sum_axis)
    true_sum = count_nonzero(y_true, axis=sum_axis)

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = _prf_divide(tp_sum, pred_sum)
        recall = _prf_divide(tp_sum, true_sum)
        f_score = (2 * precision * recall / (1 * precision + recall))
        f_score[tp_sum == 0] = 0.0

    return f_score


def _prf_divide(numerator, denominator):
    result = numerator / denominator
    mask = denominator == 0.0
    if not np.any(mask):
        return result
    # remove infs
    result[mask] = 0.0
    return result
