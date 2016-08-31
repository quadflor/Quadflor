# Quadflor

## Description

This is a text-processing pipeline for multi-label classification of scientific
documents and its evaluation.  Given a domain specific thesaurus, the different
algorithms learn how to assign its labels to documents. It also supports
optional concept extraction, synset resolution, spreading activation, and
hierarchical weighting.  The most notable contribution is a stacked classifier
called `LRDT`, which consists of stochastic gradient descent (optimizing
logistic regression) and decision trees.

## Using the example

An example call with tfidf features and stochastic gradient descent:

    ```./run.py -tf sgd -k Code/lucid_ml/file_paths.json -Kexample-titles -i```

where `file_paths.json` should contain the key given by `-K` specifying the
paths to data (`X`), the gold standard (`y`), and the thesaurus (`thes`).
