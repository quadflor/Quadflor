import json
import os
from tempfile import NamedTemporaryFile
from unittest import TestCase

import run


class TestIntegration(TestCase):
    # def test_ctf_idf(self):
    #     # data
    #     def options():
    #         pass
    #     parser = run.generate_parser()
    #     default_values = parser.get_default_values().__dict__
    #     for default in default_values.items():
    #         setattr(options, default[0], default[1])
    #     options.clf_key = 'brknnb'
    #     options.persist = True
    #     options.k = 500
    #     options.concepts = True
    #     options.terms = True
    #     options.one_fold = True
    #     run.run(options)

    def setUp(self):
        docs = NamedTemporaryFile(delete=False, mode='w')
        print('id0,|this is a fancy title|', file=docs)
        print('id1,|fancy title this is|', file=docs)
        print('id2,|not at at all pretty|', file=docs)
        print('id3,|heading this is not|', file=docs)
        self.doc_name = docs.name
        docs.seek(0)
        docs.close()
        gold = NamedTemporaryFile(delete=False, mode='w')
        self.gold_path = gold.name
        print('id0,1999,lid0,lid1', file=gold)
        print('id1,1999,lid0,lid1', file=gold)
        print('id2,1999,lid2,lid3', file=gold)
        print('id3,1999,lid2,lid4', file=gold)
        gold.close()
        self.thesaurus = {'00': {'prefLabel': ['fancy'], 'broader': ['0b'], 'related': ['0r'],
                                 'narrower': ['01'], 'altLabel': []},
                          # TODO: consider: this is removed by lemmatization.
                          '01': {'prefLabel': ['not at all'], 'broader': ['00'], 'related': ['1r'],
                                 'narrower': ['02','03'], 'altLabel': ['at all']},
                          '02': {'prefLabel': ['pretty'], 'broader': ['01'], 'related': ['2r'],
                                 'narrower': ['03'], 'altLabel': ['beautiful']},
                          '03': {'prefLabel': ['heading'], 'broader': ['01','02'], 'related': ['2r'],
                                 'narrower': [], 'altLabel': []},
                          }
        with NamedTemporaryFile(delete=False, mode ='w', suffix='json') as thes:
            json.dump(self.thesaurus, thes)
            self.thes_name = thes.name

    def tearDown(self):
        os.remove(self.doc_name)
        os.remove(self.gold_path)
