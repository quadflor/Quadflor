from sklearn.base import BaseEstimator, TransformerMixin

from utils.nltk_normalization import NltkNormalizer

import numpy as np




class TextEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, 
                 tokenize = NltkNormalizer().split_and_normalize, 
                 input_format = "content", 
                 max_words = None, 
                 pretrained = None,
                 restrict_pretrained = True):
        
        self.tokenize = tokenize
        self.input = input_format
        self.max_words = max_words
        self.pretrained = pretrained
        self.restrict_pretrained = restrict_pretrained
    
    def _maybe_load_text(self, text):
        if self.input == "filename":
            with open(text, 'r') as text_file:
                text = text_file.read()
        
        return text
    
    def _limit_num_words(self, words, max_length):
        if self.max_words is not None:
            return words[:max_length]
        else:
            return words
    
    @staticmethod
    def _load_pretrained_vocabulary(filename, word_restrictions):
        mapping = {}
        
        with open(filename + ".tmp", 'w') as temp_embedding_file:
        
            with open(filename,'r') as embedding_file:
                embedding_size = int(embedding_file.readline().strip().split(" ")[1])
    
                i = 0
                for line in embedding_file.readlines():
                    row = line.strip().split(' ')
    
                    # make sure we dont use escape sequences and so on
                    if len(row) == embedding_size + 1:
                        if row[0] in mapping:
                            print(row[0], "is already in mapping")
                        elif word_restrictions is None or row[0] in word_restrictions:
                            mapping[row[0]] = i
                            i += 1
                            temp_embedding_file.write(line)
        return mapping, i
    
    def _extract_words(self, text):
        # if full-text: load it first
        text = self._maybe_load_text(text)  
                
        # tokenize training text
        words = self.tokenize(text)
        words = self._limit_num_words(words, self.max_words)
        
        return words
    
    def _set_of_all_words(self, X):
        all_words = set()
        for text in X:
            all_words.update(set(self._extract_words(text)))
            
        return all_words
            
    def fit(self, X, y = None):
        """
        X is expected to be a list of either strings, which represent the text directly, or paths to the
        file where to find the text.
        """
        
        if self.pretrained is None:
            mapping = {}
            max_index = 1
            max_length = 0
            for text in X:
                
                words = self._extract_words(text)
            
                # build mapping from word to index
                
                for word in words:
                    if word not in mapping:
                        mapping[word] = max_index
                        max_index += 1
                
                # determine maximum length of a text for padding
                if len(words) > max_length:
                    max_length = len(words)
                    
        
        else:
            
            if self.restrict_pretrained:
                all_words = self._set_of_all_words(X)
            else:
                all_words = None
            
            mapping, max_index = TextEncoder._load_pretrained_vocabulary(self.pretrained, all_words)
            max_length = 0
            for text in X:
                
                words = self._extract_words(text)
                
                if len(words) > max_length:
                    max_length = len(words)
        
        # save variables required for transformation step
        self.mapping = mapping
        self.max_index = max_index - 1
        self.max_length = max_length
        
        return self
    
    def transform(self, X, y = None):
        
        encoding_matrix = np.zeros((len(X), self.max_length), dtype = np.int32) 
        for i, text in enumerate(X):
            
            text = self._maybe_load_text(text)
            
            # tokenize test text
            words = self.tokenize(text)
            # make sure we do not exceed the maximum length from training samples
            words = self._limit_num_words(words, self.max_length)
            
            # apply mapping from word to integer
            id_sequence = np.array([self.mapping[word] for word in words if word in self.mapping])
            
            encoding_matrix[i, :len(id_sequence)] = id_sequence
        
        max_index_column = np.zeros((len(X), 1), dtype = np.int32)
        max_index_column.fill(self.max_index)
        return np.hstack((encoding_matrix, max_index_column))
