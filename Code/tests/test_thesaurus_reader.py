import json
import os
import unittest
from collections import OrderedDict
from tempfile import NamedTemporaryFile
from unittest import TestCase
import networkx as nx
from lucid_ml.utils.thesaurus_reader import ThesaurusReader


class ThesaurusReaderTest(TestCase):
    def setUp(self):
        self.thes_path = make_thes()
        self.maxDiff = None
        unittest.util._MAX_LENGTH = 2000

    def tearDown(self):
        os.remove(self.thes_path)

    def test_mappings(self):
        tr = ThesaurusReader(self.thes_path)
        tr._thesaurus = OrderedDict(
                [('root', {'narrower': ['topc'], 'broader': [], 'prefLabel': ['rtl'], 'altLabel': []}),
                 ('topc', {'narrower': ['636688', '636686'], 'broader': ['root'],
                           'prefLabel': ['Reduction'],
                           'altLabel': ['Alt Reduction']})])
        self.assertEquals({'rtl': 0, 'Reduction': 1, 'Alt Reduction': 1}, tr.vocabulary)
        self.assertEquals({'root': 0, 'topc': 1}, tr.nodename_index)
        self.assertEquals({0: 'root', 1: 'topc'}, tr.index_nodename)
        # thesaurus = setify(tr.thesaurus)
        # expected = setify(expected)
        # self.assertEquals(thesaurus, expected)

    def test_create_with_rdf_and_get_thesaurus_dict(self):
        tr = ThesaurusReader(self.thes_path)
        expected = {
            'root': {'narrower': ['topc'], 'broader': [], 'prefLabel': ['root'], 'altLabel': []},
            'topc': {'narrower': ['636688', '636686'], 'broader': ['root'], 'prefLabel': ['reduction'],
                     'altLabel': ['alternative reduction']},
            '636688': {'narrower': [], 'broader': ['topc'], 'prefLabel': ['reduction emission'],
                       'altLabel': []},
            '636686': {'narrower': [], 'broader': ['topc'], 'prefLabel': ['security risk'], 'altLabel': []}
        }

        thesaurus = setify(tr.thesaurus)
        expected = setify(expected)
        self.assertEquals(thesaurus, expected)

    def test_get_as_networkx(self):
        tr = ThesaurusReader(self.thes_path)
        tr._thesaurus = OrderedDict(
                [('root', {'narrower': ['topc'], 'broader': [], 'prefLabel': ['rtl'], 'altLabel': []}),
                 ('topc', {'narrower': [], 'broader': ['root'],
                           'prefLabel': ['Reduction'],
                           'altLabel': ['Alt Reduction']})])
        expected = nx.DiGraph()
        expected.add_node(0)
        expected.add_node(1)
        expected.add_edge(0, 1, weight=1)
        self.assertEquals(expected.nodes(data=True), tr.nx_graph.nodes(data=True))
        self.assertEquals(expected.edges(data=True), tr.nx_graph.edges(data=True))

    def test_nx_root(self):
        tr = ThesaurusReader(self.thes_path)
        tr._thesaurus = OrderedDict(
                [('root', {'narrower': ['topc'], 'broader': [], 'prefLabel': ['rtl'], 'altLabel': []}),
                 ('topc', {'narrower': [], 'broader': ['root'],
                           'prefLabel': ['Reduction'],
                           'altLabel': ['Alt Reduction']})])
        self.assertEquals(0, tr.nx_root)

    def test_read_json(self):
        graph = {
            'root': {'narrower': ['topc'], 'broader': [], 'prefLabel': [], 'altLabel': []},
            'topc': {'narrower': ['636688', '636686'], 'broader': ['root'], 'prefLabel': ['reduction'],
                     'altLabel': ['alt reduction']},
            '636688': {'narrower': [], 'broader': ['topc'], 'prefLabel': ['reduction emission'],
                       'altLabel': ["bullshit label deprecated", "bullshit label"]},
            '636686': {'narrower': [], 'broader': ['topc'], 'prefLabel': ['security risk'], 'altLabel': []}
        }
        with NamedTemporaryFile(delete=False, suffix='.json', mode='w') as temp:
            json.dump(graph, temp)
        tr = ThesaurusReader(temp.name)
        thesaurus = setify(tr.thesaurus)
        expected = setify(graph)
        self.assertEquals(thesaurus, expected)

    def test_normalize(self):
        tr = ThesaurusReader(self.thes_path)
        tr._thesaurus = OrderedDict(
                [('root', {'narrower': [], 'broader': [], 'prefLabel': ['sr word'],
                           'altLabel': ['A.01.C nice words', 'and or Nice WORD']})
                 ])
        expected = {'root': {'narrower': [], 'broader': [], 'prefLabel': ['word'],
                             'altLabel': ['nice word', 'nice word']}}
        tr.normalize_thesaurus()
        self.assertEquals(tr.thesaurus, expected)

    def test_nx_graph_missing_narrower(self):
        tr = ThesaurusReader(self.thes_path)
        tr._thesaurus = OrderedDict(
                [('root', {'narrower': [], 'broader': [], 'prefLabel': ['rtl'], 'altLabel': []}),
                 ('topc', {'narrower': [], 'broader': ['root'],
                           'prefLabel': ['Reduction'],
                           'altLabel': ['Alt Reduction']})])
        expected = nx.DiGraph()
        expected.add_node(0)
        expected.add_node(1)
        expected.add_edge(0, 1, weight=1)
        self.assertEquals(expected.nodes(data=True), tr.nx_graph.nodes(data=True))
        self.assertEquals(expected.edges(data=True), tr.nx_graph.edges(data=True))

    def test_only_labels_and_relations(self):
        thes_text = """
        <http://fiv-iblk.de/descriptor/1> <http://www.w3.org/2004/02/skos/core#prefLabel>  "AAAA"@en .
        <http://fiv-iblk.de/descriptor/2> <http://www.w3.org/2004/02/skos/core#prefLabel>  "BBBB"@en .
        <http://fiv-iblk.de/descriptor/3> <http://www.w3.org/2004/02/skos/core#prefLabel>  "CCCC"@en .

        <http://fiv-iblk.de/descriptor/1> <http://www.w3.org/2004/02/skos/core#narrower> <http://fiv-iblk.de/descriptor/2> .
        <http://fiv-iblk.de/descriptor/2> <http://www.w3.org/2004/02/skos/core#broader> <http://fiv-iblk.de/descriptor/1> .

        <http://fiv-iblk.de/descriptor/1> <http://www.w3.org/2004/02/skos/core#narrower> <http://fiv-iblk.de/descriptor/3> .
        <http://fiv-iblk.de/descriptor/3> <http://www.w3.org/2004/02/skos/core#broader> <http://fiv-iblk.de/descriptor/1> .
        """
        tr = ThesaurusReader(fill_file(thes_text))
        expected = setify({
            '1': {'narrower': ['3', '2'], 'broader': [], 'prefLabel': ['aaaa'], 'altLabel': []},
            '2': {'narrower': [], 'broader': ['1'], 'prefLabel': ['bbbb'], 'altLabel': []},
            '3': {'narrower': [], 'broader': ['1'], 'prefLabel': ['cccc'], 'altLabel': []},
        })
        self.assertEquals(setify(tr.thesaurus), expected)

    def test_slash_in_labels(self):
        thes_text = """
        <http://fiv-iblk.de/descriptor/1> <http://www.w3.org/2004/02/skos/core#prefLabel>  "AAA/XXX"@en .
        <http://fiv-iblk.de/descriptor/2> <http://www.w3.org/2004/02/skos/core#prefLabel>  "BBA/XXX"@en .
        <http://fiv-iblk.de/descriptor/3> <http://www.w3.org/2004/02/skos/core#prefLabel>  "CCA/XXX"@en .

        <http://fiv-iblk.de/descriptor/1> <http://www.w3.org/2004/02/skos/core#narrower> <http://fiv-iblk.de/descriptor/2> .
        <http://fiv-iblk.de/descriptor/2> <http://www.w3.org/2004/02/skos/core#broader> <http://fiv-iblk.de/descriptor/1> .

        <http://fiv-iblk.de/descriptor/1> <http://www.w3.org/2004/02/skos/core#narrower> <http://fiv-iblk.de/descriptor/3> .
        <http://fiv-iblk.de/descriptor/3> <http://www.w3.org/2004/02/skos/core#broader> <http://fiv-iblk.de/descriptor/1> .
        """
        tr = ThesaurusReader(fill_file(thes_text))
        expected = setify({
            '1': {'narrower': ['3', '2'], 'broader': [], 'prefLabel': ['aaa xxx'], 'altLabel': []},
            '2': {'narrower': [], 'broader': ['1'], 'prefLabel': ['bba xxx'], 'altLabel': []},
            '3': {'narrower': [], 'broader': ['1'], 'prefLabel': ['cca xxx'], 'altLabel': []},
        })
        self.assertEquals(setify(tr.thesaurus), expected)


def setify(dic):
    for i in dic.items():
        for j in i[1].items():
            dic[i[0]][j[0]] = set(j[1])
    return dic


def make_thes():
    thes_text = """
    <http://fiv-iblk.de/descriptor/636688> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2004/02/skos/core#Concept> .
    <http://fiv-iblk.de/descriptor/636686> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2004/02/skos/core#Concept> .
    <http://fiv-iblk.de/descriptor/topc> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2004/02/skos/core#Concept> .
    <http://fiv-iblk.de/descriptor/636688> <http://www.w3.org/2004/02/skos/core#prefLabel> "Emissionsreduktion"@de .
    <http://fiv-iblk.de/descriptor/636688> <http://www.w3.org/2004/02/skos/core#prefLabel> "Reduction of emissions"@en .
    <http://fiv-iblk.de/descriptor/636686> <http://www.w3.org/2004/02/skos/core#prefLabel> "Sicherheitsrisiken"@de .
    <http://fiv-iblk.de/descriptor/636686> <http://www.w3.org/2004/02/skos/core#prefLabel> "Security risks"@en .
    <http://fiv-iblk.de/descriptor/636688> <http://www.w3.org/2004/02/skos/core#broader> <http://fiv-iblk.de/descriptor/topc> .
    <http://fiv-iblk.de/descriptor/636686> <http://www.w3.org/2004/02/skos/core#broader> <http://fiv-iblk.de/descriptor/topc> .
    <http://fiv-iblk.de/descriptor/topc> <http://www.w3.org/2004/02/skos/core#narrower> <http://fiv-iblk.de/descriptor/636688> .
    <http://fiv-iblk.de/descriptor/topc> <http://www.w3.org/2004/02/skos/core#narrower> <http://fiv-iblk.de/descriptor/636686> .
    <http://fiv-iblk.de/descriptor/topc> <http://www.w3.org/2004/02/skos/core#prefLabel> "Reduction"@en .
    <http://fiv-iblk.de/descriptor/topc> <http://www.w3.org/2004/02/skos/core#altLabel> "Alternative Reduction"@en .
    <http://fiv-iblk.de/descriptor/topc> <http://www.w3.org/2004/02/skos/core#prefLabel> "Reduzierung"@de .
    <http://fiv-iblk.de/descriptor/topc> <http://www.w3.org/2004/02/skos/core#topConceptOf> <http://fiv-iblk.de> .
    <http://fiv-iblk.de> <http://www.w3.org/2004/02/skos/core#hasTopConcept> <http://fiv-iblk.de/descriptor/topc> .
    <http://fiv-iblk.de/descriptor/636687> <http://www.w3.org/2000/01/rdf-schema#label>  "Bullshit label"@en .
    <http://fiv-iblk.de/descriptor/636687> <http://www.w3.org/2002/07/owl#deprecated> "true"^^<http://www.w3.org/2001/XMLSchema#boolean> .
    <http://fiv-iblk.de/descriptor/636687> <http://purl.org/dc/terms/isReplacedBy> <http://fiv-iblk.de/descriptor/636688> .
    <http://fiv-iblk.de/descriptor/636689> <http://www.w3.org/2002/07/owl#deprecated> "true"^^<http://www.w3.org/2001/XMLSchema#boolean> .
    <http://fiv-iblk.de/descriptor/636689> <http://purl.org/dc/terms/isReplacedBy> <http://fiv-iblk.de/descriptor/636687> .
    <http://fiv-iblk.de/descriptor/636689> <http://www.w3.org/2000/01/rdf-schema#label> "Bullshit label Deprecated"@en .
    """
    return fill_file(thes_text)


def fill_file(thes_text):
    with NamedTemporaryFile(delete=False, suffix='.nt', mode='w') as f:
        print(thes_text, file=f)
        name = f.name
    return name
