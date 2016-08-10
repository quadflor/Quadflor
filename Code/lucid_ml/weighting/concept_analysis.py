import csv
import datrie as tr
import os
import string

import atexit
from sklearn.base import BaseEstimator

from utils.nltk_normalization import NltkNormalizer

_alphabet = set(string.ascii_lowercase + string.digits + ' ')
counter = 0


class ConceptAnalyzer(BaseEstimator):
    """Find concepts using thesaurus. Scans data with a window of the size of the longest label in the
    thesaurus. When finding more than one matches in the same window, it takes the longest one.

    Parameters
    ----------
    thesaurus: dict[str,dict[str,list[str]]]
        Thesaurus as dict with format:{'id': {'prefLabel': [],'broader': [],'narrower': [],'altLabel': []}, ...}
        As returned by ThesaurusReader.
    input: str default='content'
        One of {'filename', 'content'}.
        If 'filename', the sequence passed as an argument to fit is expected to be a list of filenames that need
        reading to fetch the raw content to analyze.
        Otherwise the input is expected to be the sequence strings items are expected to be analyzed directly.

    """

    # noinspection PyShadowingBuiltins
    def __init__(self, thesaurus, input='content', persist=False, persist_dir='./persistence', file_path=None,
                 repersist=False):
        if persist and not input == 'filename':
            print('Can only persist concepts separately when reading files.')
        self.persist_dir = persist_dir
        self.persist = persist and input == 'filename'
        self.input = input
        self.thesaurus = thesaurus
        self.normalizer = NltkNormalizer()
        self.reverse_thesaurus, self.length = self._construct_reverse_thesaurus()
        if self.persist:
            self.save_path = os.path.join(self.persist_dir, os.path.abspath(file_path).replace(os.sep, ''))
            self.is_saved = not repersist and os.path.isfile(self.save_path)
            if self.is_saved:
                self.persistence_file = open(self.save_path, mode='r')
                print('Reading persisted concepts from ' + self.save_path)
                self.doc_entities = self._read_saved()
            else:
                self.persistence_file = open(self.save_path, mode='w')
                print('Persisting concepts to: ' + self.save_path)
                self.writer = csv.writer(self.persistence_file, delimiter='\t')
            atexit.register(self.close_persistence_file)

    def close_persistence_file(self):
        if self.persistence_file:
            self.persistence_file.close()

    def __repr__(self):
        return self.__class__.__name__

    def analyze(self, doc):
        if self.persist and self.is_saved:
            return self.doc_entities[doc]
        if self.input == 'filename':
            with open(doc) as f:
                entities = self._analyze(f.read())
                if self.persist:
                    self.writer.writerow([doc] + entities)
                return entities
        analyze = self._analyze(doc)
        return analyze

    def _analyze(self, doc):
        lemmas = self.normalizer.split_and_normalize(doc)
        return self._get_entities(lemmas)

    def _construct_reverse_thesaurus(self):
        reverse_thesaurus = tr.Trie(_alphabet)
        longest_entity_length = 0
        for entry in self.thesaurus.items():
            entity_id = entry[0]
            label_types = set.intersection({'prefLabel', 'altLabel'}, set(entry[1].keys()))
            label_lists = [entry[1][label_type] for label_type in label_types]
            labels = [item for sublist in label_lists for item in sublist]
            for label in labels:
                reverse_thesaurus[label] = entity_id
                length = len(label.split(' '))
                if length > longest_entity_length:
                    longest_entity_length = length
        return reverse_thesaurus, longest_entity_length

    def _get_entities(self, lemmas):
        r = []
        while lemmas:
            first_words = ' '.join(lemmas[:self.length])
            candidates = self.reverse_thesaurus.prefix_items(first_words)
            num_remove = 1
            if candidates:
                longest_candidate = candidates[-1][1]
                num_remove = longest_candidate.count(' ') + 1
                r.append(longest_candidate)
            del lemmas[:num_remove]
        return r

    def _read_saved(self):
        file_entities = {}
        reader = csv.reader(self.persistence_file, delimiter='\t')
        for row in reader:
            key = row[0]
            entities = row[1:]
            file_entities[key] = entities
        return file_entities
