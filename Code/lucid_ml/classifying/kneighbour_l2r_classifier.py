from sklearn.neighbors import NearestNeighbors, LSHForest
import numpy as np
from sklearn.base import BaseEstimator
import subprocess
from scipy.sparse.csr import csr_matrix
from nltk.translate import IBMModel1
from nltk.translate import AlignedSent
import sys
import scipy.sparse as sps
from sklearn.externals.joblib import Parallel, delayed


class KNeighborsL2RClassifier(BaseEstimator):
    """
    This classifier first examines the labels that appear in the n-nearest documents and then generates
    features for each of the labels. Given these features, a list-wise ranking algorithm then ranks the labels.
    Finally, the k highest ranked labels are assigned as output.
    
    The features that are generated are the following:
        - the sum of the cosine similarities to all the documents in the n-neighborhood that the label is actually assigned to
        - the total count of appearances of the label in the document
        - the probability that the label translates to the document's title according to the IBM1-translation-model.
        - the count of how often the label directly appears in the title
        
    This algorithm uses the implementation from [RankLib] in the background. Therefore, it assumes a Java-
    installation to be available and the RankLib-2.5.jar file to be in the same folder as this file.
        
    The whole procedure's main idea was taken from [MLA]. However, as some of their techniques cannot be applied
    in the same way to our application, some of the implementation details differ.
    
    Note that, as opposed to the original paper, we need to decide which labels from the ranking to assign to the
    document. In our approach, we take the top k elements from each list where k equals the average number of labels
    assigned to a document in the training corpus rounded to an integer.
    
    References
    ----------
    [MLA]   Huang, Minlie, Aurélie Névéol, and Zhiyong Lu. 
            "Recommending MeSH terms for annotating biomedical articles." 
            Journal of the American Medical Informatics Association 18.5 (2011): 660-667.
    [RankLib] https://sourceforge.net/p/lemur/wiki/RankLib/
    
    Parameters
    ----------
    n_jobs: int, default=1
        Number of jobs to use for extracting the training / test sets for l2r.
    n_neighbors: int, default=20
        The size of the neighborhood, from which labels are considered as input to the ranking algorithm
    max_iterations: int, default=300
        Given as input to the RankLib library. Determines the number of iterations to train the ranking classifier with.
    count_concepts: boolean, default=False
        Must indicate whether the feature vector of a document contains concepts or not.
    number_of_concepts: boolean, default=0
        Must indicate the number of concepts contained in the document's feature vector.
    count_terms: terms, default=True
        Must indicate whether the feature vector of a document contains terms or not.
    training_validation_split: float, default = 1.0
        Determines how much of the training data passed to fit() is used for training. Rest is used for validation.
    algorithm_id: string, default = '7'
        Specifies the id of the (list-wise) L2R algorithm to apply. Must be either '3' for AdaRank, '4' for Coordinate Ascent, '6' for LambdaMART, or '7' for ListNET.
    """
    def __init__(self, use_lsh_forest=False, n_neighbors=20, max_iterations = 300, count_concepts = False, number_of_concepts = 0,
                count_terms = False, training_validation_split = 0.8, algorithm_id = '7', l2r_metric = "ERR@k", n_jobs = 1, translation_probability = False, **kwargs ):
        
        self.n_neighbors = n_neighbors
        self.knn = LSHForest(n_neighbors=n_neighbors, **kwargs) if use_lsh_forest else NearestNeighbors(
            n_neighbors=n_neighbors, **kwargs)
        self.y = None
        self.max_iterations = max_iterations
        self.count_concepts = count_concepts
        self.count_terms = count_terms
        self.number_of_concepts = number_of_concepts
        self.training_validation_split = training_validation_split
        self.algorithm_id = algorithm_id
        self.l2r_metric = l2r_metric
        self.n_jobs = n_jobs
        self.translation_probability = translation_probability

    def fit(self, X, y):
        self.y = y
        self.knn.fit(X)
        
        if sps.issparse(y):
            average_labels = int(np.round(np.mean(y.sum(axis = 1)), 0))
        else:
       	    average_labels = int(np.round(np.mean(np.sum(y, axis = 1)), 0))
        self.topk = average_labels
        
        distances_to_neighbors,neighbor_id_lists = self.knn.kneighbors()
        
        if self.translation_probability:
            self._train_translation_model(X,y)
        
        # create features for each label in a documents neighborhood and write them in a file
        self._extract_and_write(X, neighbor_id_lists, distances_to_neighbors, y = y)
        # use the RankLib library algorithm to create a ranking model
        subprocess.call(["java", "-jar", "RankLib-2.5.jar", "-train", "l2r_train", "-tvs", str(self.training_validation_split), 
                         "-save", "l2r_model", "-ranker" , self.algorithm_id, "-metric2t", self.l2r_metric, "-epoch", str(self.max_iterations),
                         "-norm", "zscore"])


    def _train_translation_model(self, X, y):
        translations = []
        
        for row in range(0, X.shape[0]):
            
            title = _recompose_title(X, row)
            for label in _get_labels_of_row(row, y):
                translations.append(AlignedSent(title, [label]))
        
    
        ibm1 = IBMModel1(translations, 5)
        self.ibm1 = ibm1
    
    def _extract_and_write(self, X, neighbor_id_lists, distances_to_neighbors, fileName = "l2r_train", y = None):
        
        labels_in_neighborhood = Parallel(n_jobs=self.n_jobs)(
            delayed(_create_training_samples)(cur_doc, neighbor_list, X, y, cur_doc + 1, distances_to_neighbors, 
                                              self.count_concepts, self.count_terms, self.number_of_concepts, 
                                              self.ibm1 if self.n_jobs == 1 and self.translation_probability else None) for cur_doc, neighbor_list in enumerate(neighbor_id_lists))
            
            
        doc_to_neighborhood_dict = self._merge_dicts(labels_in_neighborhood)
        
        filenames = ["samples_" + str(qid + 1) + ".tmp" for qid in range(len(doc_to_neighborhood_dict))]
        with open(fileName, 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
                outfile.write('\n')
                
        return doc_to_neighborhood_dict
    
    def _merge_dicts(self, labels_in_neighborhood):
        neighborhood_dict = {}
        for i, labels in enumerate(labels_in_neighborhood):
            neighborhood_dict[str(i + 1)] = labels
            
        return neighborhood_dict
                         
    def _intersection_percent(self, all_labels, curr_labels):
        return len(all_labels.intersection(curr_labels)) / len(curr_labels)
    
    def _predict_scores(self, X):
        distances, neighbor_id_lists = self.knn.kneighbors(X, n_neighbors=self.n_neighbors)
        doc_to_neighborhood_dict = self._extract_and_write(X, neighbor_id_lists, distances, fileName="l2r_test", y = self.y)
        
        # use the created ranking model to get a ranking
        subprocess.call(["java", "-jar", "RankLib-2.5.jar", "-rank", "l2r_test", "-score", "test_scores", "-load", "l2r_model", "-norm", "zscore"])
        
        scoresFile = open("test_scores", "r")
        
        results = {}
        for line in scoresFile:
            qid, score = self._extract_score(line)
            if str(qid) in results:
                labels = results[str(qid)]
                labels.append(score)
                results[str(qid)] = labels
            else:
                results[str(qid)] = [score]
        
        return self._extract_topk_score(doc_to_neighborhood_dict, results)
    
    def predict_proba(self, X):
        """ we set the probability of all labels we do not observe in the neighborhood at all to 0 
        (or rather to the smallest float value possible to be safe, because we don't normalize).
        Otherwise, the probability of a label corresponds to the score assigned by the ranking algorithm.
        """ 
        top_k_save = self.topk
        self.topk = self.y.shape[1]
        
        doc_to_neighborhood_dict = self._predict_scores(X)
        
        self.topk = top_k_save
        
        probabilities = np.zeros((X.shape[0], self.y.shape[1]))
        probabilities.fill(sys.float_info.min)
        
        for i in range(0,X.shape[0]):
            for label, score in doc_to_neighborhood_dict[str(i + 1)]:
                probabilities[i, label] = score
                
        return probabilities
        
    def predict(self, X):
        predictions = csr_matrix((X.shape[0], self.y.shape[1]))
        doc_to_neighborhood_dict = self._predict_scores(X)
        
        for i in range(0,X.shape[0]):
            for label, _ in doc_to_neighborhood_dict[str(i + 1)]:
                predictions[i, label] = 1
        return predictions
        
    def _extract_score(self, line):
        entries = line.split()
        qid = entries[0]
        score = entries[2]
        return (qid, score)
    
    def _extract_topk_score(self, doc_to_neighborhood_dict, results):
        for key, value in results.items():
            top_indices = self._get_top_indices(value)
            top_values = self._get_top_values(value)
            labels = doc_to_neighborhood_dict[key]
            toplabels = [(labels[index], top_value) for index, top_value in zip(top_indices, top_values)] 
            doc_to_neighborhood_dict[key] = toplabels
            
        return doc_to_neighborhood_dict
    
    def _get_top_indices(self, array):
        return self._maxEntries(array, self.topk)
    
    def _get_top_values(self, array):
        return self._maxValues(array, self.topk)
    
    # this one was stolen here: http://stackoverflow.com/questions/12787650/finding-the-index-of-n-biggest-elements-in-python-array-list-efficiently
    def _maxEntries(self,a,N):
        return np.argsort(a)[::-1][:N]
    
    def _maxValues(self,a,N):
        return np.sort(a)[::-1][:N]
    

def _create_training_samples(curr_doc, neighbor_list, X, y, qid, distances_to_neighbors, count_concepts, count_terms, number_of_concepts, ibm1):
    l2r_training_samples = open("samples_" + str(qid) + ".tmp", "w", encoding='utf-8')    
    labels_in_neighborhood = y[neighbor_list]
            
    labels_per_neighbor = []
    for i in range(0,labels_in_neighborhood.shape[0] - 1):
        cols = _get_labels_of_row(i,labels_in_neighborhood)
        labels_per_neighbor.append(cols)
        
    all_labels_in_neighborhood = [label for labels in labels_per_neighbor for label in labels]
    all_labels_in_neighborhood = list(set(all_labels_in_neighborhood))
        
    for label in all_labels_in_neighborhood:
        l2r_training_samples.write(_generate_line_for_label(label, qid, X, y, curr_doc, labels_per_neighbor, neighbor_list,
                                                            distances_to_neighbors[curr_doc], count_concepts, count_terms, number_of_concepts, ibm1))
    
    return all_labels_in_neighborhood

def _get_labels_of_row(i,matrix):
        labels_of_single_neighbor = matrix[i]
        rows, cols = labels_of_single_neighbor.nonzero()
        return cols
    
# we are generating a line per label according to the notation given in http://www.cs.cornell.edu/People/tj/svm_light/svm_rank.html            
def _generate_line_for_label(label, qid, X, y, curr_doc, labels_per_neighbor, neighbor_list, distances_to_neighbor, count_concepts, count_terms, number_of_concepts, ibm1):
    # set the target value to 1 (i.e., the label is ranked high), if the label is actually assigned to the current document, 0 otherwise
    if(y != None): 
        label_in_curr_doc = 1 if label in _get_labels_of_row(curr_doc, y) else 0
    else:
        label_in_curr_doc = 0
    
    features = _extract_features(label, curr_doc, labels_per_neighbor, neighbor_list, distances_to_neighbor, X, y, count_concepts, count_terms, number_of_concepts, ibm1)
    featureString = " ".join([str(i + 1) + ":" + str(val) for i,val in enumerate(features)])
    #if y != None:
    line_for_label = str(label_in_curr_doc) + " "
    #else:
                
    line_for_label = line_for_label + "qid:" + str(qid) + " " + featureString +  "\n"
    return line_for_label

def _extract_features(label, curr_doc, labels_per_neighbor, neighbor_list, distances_to_neighbor, X, y, count_concepts, count_terms, number_of_concepts, ibm1):
    sum_similarities = 0
    for i, neighbor_list in enumerate(labels_per_neighbor):
        if label in neighbor_list:
            sum_similarities += distances_to_neighbor[i]
    
    count_in_neighborhood = [some_label for labels in labels_per_neighbor for some_label in labels].count(label)
    
    title = _recompose_title(X, curr_doc)
    
    if not ibm1 is None:
        translation_probability = 0
        for word in title:
            translation_probability += ibm1.translation_table[word][str(label)]
        
        features = [count_in_neighborhood, sum_similarities, translation_probability]
    else:
        features = [count_in_neighborhood, sum_similarities]
    
    if count_concepts and not count_terms:
        features.append(1 if X[curr_doc,label] > 0 else 0)
    elif count_concepts and count_terms:
        features.append(1 if X[curr_doc, X.shape[1] - number_of_concepts + label] > 0 else 0)
    
    return features

def _recompose_title(X, row):
    title = []
    for word in X[row].nonzero():
        title.append(str(word))
    
    return title
