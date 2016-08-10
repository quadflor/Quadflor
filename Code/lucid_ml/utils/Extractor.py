# -*- coding: utf-8 -*-
import csv
import json
import os
import random

from utils.thesaurus_reader import ThesaurusReader

# paths to required resources:
# [0: titles/documents, 1: gold, 2: thesaurus]
def load_dataset(DATA_PATHS, key='econ62k', fulltext=False):
    if fulltext:
        data = load_documents(DATA_PATHS[key]['X'])
    else:
        data = load_titles(DATA_PATHS[key]['X'])
    gold = load_gold(DATA_PATHS[key]['y'])
    data_list, gold_list = reduce_dicts(data, gold)
    tr = ThesaurusReader(DATA_PATHS[key]['thes'])

    return data_list, gold_list, tr


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


def reduce_dicts(titles, gold, shuffle=False):
    """ reduce 2 dictionaries to 2 lists,
    providing 'same index' iff 'same key in the dictionary'
    """
    titles, gold = dict(titles), dict(gold)
    titles_list = []
    gold_list = []
    for key in titles:
        titles_list.append(titles[key])
        gold_list.append(gold[key])

    if shuffle:
        # this is the only way to shuffle them
        zipped = list(zip(titles_list, gold_list))
        random.shuffle(zipped)
        titles_list, gold_list = zip(*zipped)

    return titles_list, gold_list
