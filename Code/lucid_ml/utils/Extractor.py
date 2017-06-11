# -*- coding: utf-8 -*-
import csv
import json
import os
import random
import pandas as pd
import math

from utils.thesaurus_reader import ThesaurusReader

def split_labels(labels_string):
    return labels_string.split(",")

def load_data(df, fulltext, fixed_folds):
    
    if fulltext:
        content_column = "fulltext_path"
    else:
        content_column = "title"
        
    # bring title/fulltext in dictionary format    
    content = dict()
    for row in df.iterrows():
        content[row[1].loc["id"]] = row[1][content_column]

    # bring goldstandard in dictionary format
    gold = dict()
    for row in df.iterrows():
        gold[row[1]["id"]] = split_labels(row[1]["labels"])
        
    # by default, there is only one fold
    folds = dict()
    folds.update({key : 0 for key in gold}) 
    if fixed_folds:
        for row in df.iterrows():
            folds[row[1]["id"]] = int(row[1]["fold"])

    data_list, gold_list, fold_list = reduce_dicts([content, gold, folds])
    
    if fulltext:
        fulltext_indices = [index for index, x in enumerate(data_list) if type(x) == str or not math.isnan(x)]
        
        def elems_by_indices(some_list):
            return [some_list[i] for i in fulltext_indices]
        
        data_list, gold_list, fold_list = elems_by_indices(data_list), elems_by_indices(gold_list), elems_by_indices(fold_list)
        print(len(data_list))

    return data_list, gold_list, fold_list

# paths to required resources:
# [0: titles/documents, 1: gold, 2: thesaurus]
def load_dataset(DATA_PATHS, key='econ62k', fulltext=False, fixed_folds = False):
    data_set = DATA_PATHS[key]

    dataset_format = data_set["format"] if "format" in data_set else "separate"

    if dataset_format == "separate":
    
        if fulltext:
            data = load_documents(DATA_PATHS[key]['X'])
        else:
            data = load_titles(DATA_PATHS[key]['X'])
        gold = load_gold(DATA_PATHS[key]['y'])
        data_list, gold_list = reduce_dicts(data, gold)
        tr = ThesaurusReader(DATA_PATHS[key]['thes'])

        return data_list, gold_list, tr

    elif dataset_format == "combined":
        # extract the available folds and keep each folds samples in a separate list
        df = pd.read_csv(data_set["X"])
        data_list, gold_list, fold_list = load_data(df, fulltext, fixed_folds)
        tr = None if "thes" not in data_set else ThesaurusReader(data_set['thes'])
        return data_list, gold_list, tr, fold_list

    else:
        print("Format not recognized:", dataset_format)
        raise ValueError("No such format: " + dataset_format)

# expected format: <document_id>\t<title>
def load_titles(titles_path):
    titles = dict()
    with open(titles_path, 'r', errors='ignore') as f:
        rd = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in rd:
            title_id = line[0]
            title = line[1]
            titles[title_id] = title

    return titles


# expected format: <document_id>.TXT, without '.' in <document_id>
def load_documents(docs_path):
    documents = dict()
    docs_list = os.listdir(docs_path)
    for doc in docs_list:
        doc_id = doc.split('.')[0]
        documents[doc_id] = docs_path + '/' + doc

    return documents


# expected format: <document_id>\t<annotation1>\t<annotation2>\t...
def load_gold(path):
    gold = dict()
    with open(path, 'r') as f:
        rd = csv.reader(f, delimiter='\t')
        for line in rd:
            gold_id = line[0]
            g = line[1:]
            gold[gold_id] = g

    return gold


def reduce_dicts(dicts, shuffle=False):
    """ reduces a list of dictionaries to a list of lists,
    providing 'same index' iff 'same key in the dictionary'
    """
    dicts = list(map(dict, dicts))
    lists = [[] for i in range(len(dicts))]
    
    for key in dicts[0]:
        for i, some_dict in enumerate(dicts):
            lists[i].append(some_dict[key])

    if shuffle:
        # this is the only way to shuffle them
        zipped = list(zip(lists))
        random.shuffle(zipped)
        lists = zip(*zipped)

    return tuple(lists)
