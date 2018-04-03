# Using Deep Learning for Title-Based Semantic Subject Indexing to Reach Competitive Performance to Full-Text

This repository contains the code for the JCDL paper [Using Deep Learning for Title-Based Semantic Subject Indexing to Reach Competitive Performance to Full-Text](https://arxiv.org/abs/1801.06717). It is based on and extents the multi-label classification framework [Quadflor](https://github.com/quadflor/Quadflor).

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

## Replicating the results

In order to enhance the reproducability of our study, we uploaded a copy of the title datasets to Kaggle. Moreover, we provide the configurations used to produce the results from the paper.

To rerun any of the (title) experiments, do the following:
1. Download the [econbiz.csv and pubmed.csv](https://www.kaggle.com/hsrobo/titlebased-semantic-subject-indexing) files, respectively, and copy them to the folder *Resources*.
2. Open the .cfg file of the respective method that you want to run (MLP, BaseMLP, CNN, or LSTM) from the *Experiments* folder. Copy the command in the third (if you want to evaluate on a single fold) or fifth (if you want to do a full 10-fold-cross-validation) line.
4. In the command, adjust the parameter for the option --tf-model-path parameter (specifies where to save the weights of the models, which can be gigabytes, so make sure you have enough disk space), and the --pretrained_embeddings parameter to the location of the GloVe model in your environment.
5. *cd* to the folder *Code/lucid_ml* and run the command.
