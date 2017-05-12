# Quadflor

![Multi-label classification pipeline](Documentation/graphics/pipelineExtended.png)

## Description

Quadflor is a text-processing pipeline for multi-label classification of
scientific documents and its evaluation. Given a domain-specific thesaurus with
descriptor labels, the different algorithms learn how to assign these labels to
documents from a training set. The framework supports opportunities to conduct
concept extraction, synonym set resolution, spreading activation including
hierarchical re-weighting. The most notable contribution is a stacked
classifier called, which consists of stochastic gradient descent (optimizing
logistic regression) and decision trees.

## Installation

Install Python 3.4 or higher and

```sh
#install necessary packages
sudo apt-get install libatlas-base-dev gfortran python3.4-dev python3.4-venv build-essential

#install python modules in a virtual environment with pip (this may take a while):
python3 -m venv lucid_ml_environment
source lucid_ml_environment/bin/activate
cd Code
pip install -r requirements.txt
```

## Example Usage

An example call with tf-idf features and stochastic gradient descent classifier:

```sh
cd Code/lucid_ml
./run.py -tf sgd -k Code/lucid_ml/file_paths.json -K example-titles -i
```

where `file_paths.json` should contain the key given by `-K` specifying the
paths to data (`X`), the gold standard (`y`), and the thesaurus (`thes`).
