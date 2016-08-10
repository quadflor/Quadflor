import nltk
from nltk import wsd
from nltk.corpus import wordnet as wn

from utils.nltk_normalization import NltkNormalizer


class SynsetAnalyzer:
    def __init__(self):
        NltkNormalizer.install_nltk_corpora('averaged_perceptron_tagger')
        self.normalizer = NltkNormalizer()
        self.lem = nltk.WordNetLemmatizer()
        self.tagger = nltk.PerceptronTagger()
        self.translation_dict = {'J': wn.ADJ, 'N': wn.NOUN, 'R': wn.ADV, 'V': wn.VERB}

    def analyze(self, doc):
        res = []
        for sentence in self.normalizer.sent_tokenize(doc):
            tagged_sentence = self.tagger.tag(self.normalizer.split_and_normalize(sentence))
            lemmatized_doc = []
            for w, pos in tagged_sentence:
                try:
                    pos_ = pos[:1]
                    wn_postag = self.translation_dict[pos_]
                except KeyError:
                    wn_postag = None
                if wn_postag:
                    lemmatized_doc.append(self.lem.lemmatize(w, wn_postag))
            for w in lemmatized_doc:
                sense = wsd.lesk(lemmatized_doc, w)
                if sense:
                    res.append(sense.name())
        return res
