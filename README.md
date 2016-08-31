# Quadflor

## Description

This is a text-processing pipeline for multi-label classification of scientific
documents and its evaluation.  Given a domain specific thesaurus, the different
algorithms learn how to assign its labels to documents. It also supports
optional concept extraction, synset resolution, spreading activation, and
hierarchical weighting.  The most notable contribution is a stacked classifier
called ´SGDDT´, which consists of stochastic gradient descent (optimizing
logistic regression) and decision trees.

## Implementation Details

- [run.py](./Code/lucid_ml/lucid_ml/run.py)
- [BRKNeighborsClassifier](./Code/lucid_ml/lucid_ml/classifying/br_kneighbor_classifier.py.py)
- [KNeighborsListNetClassifier](./Code/lucid_ml/lucid_ml/classifying/kneighbour_listnet_classifier.py)
- [MeanCutKNeighborsClassifier](./Code/lucid_ml/lucid_ml/classifying/meancut_kneighbor_classifier.py)
- [NearestNeighbor](./Code/lucid_ml/lucid_ml/classifying/nearest_neighbor.py)
- [RocchioClassifier](./Code/lucid_ml/lucid_ml/classifying/rocchioclassifier.py)
- [ClassifierStack](./Code/lucid_ml/lucid_ml/classifying/stacked_classifier.py)
- [load\_dataset](./Code/lucid_ml/lucid_ml/utils/Extractor.py)
- [hierarchical\_f\_measure, f1\_per\_sample](./Code/lucid_ml/lucid_ml/utils/metrics.py)
- [NltkNormalizer, word\_regexp](./Code/lucid_ml/lucid_ml/utils/nltk_normalization.py)
- [Persister](./Code/lucid_ml/lucid_ml/utils/persister.py)
- [BinarySA, OneHopActivation, SpreadingActivation](./Code/lucid_ml/lucid_ml/weighting/SpreadingActivation.py)
- [SynsetAnalyzer](./Code/lucid_ml/lucid_ml/weighting/synset_analysis.py)
- [BM25Transformer](./Code/lucid_ml/lucid_ml/weighting/bm25transformer.py)
- [ConceptAnalyzer](./Code/lucid_ml/lucid_ml/weighting/concept_analysis.py)
- [GraphVectorizer](./Code/lucid_ml/lucid_ml/weighting/graph_score_vectorizer.py)
