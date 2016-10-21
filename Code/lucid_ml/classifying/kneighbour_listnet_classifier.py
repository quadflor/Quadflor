from sklearn.neighbors import NearestNeighbors, LSHForest
import numpy as np
from sklearn.base import BaseEstimator
import subprocess
from scipy.sparse.csr import csr_matrix
from nltk.translate import IBMModel1
from nltk.translate import AlignedSent


class KNeighborsListNetClassifier(BaseEstimator):
    """
    This classifier first examines the labels that appear in the n-nearest documents and then generates
    features for each of the labels. Given these features, the ranking algorithm 'ListNET' then ranks the labels.
    Finally, the k highest ranked labels are assigned as output.
    
    The features that are generated are the following:
        - the sum of the cosine similarities to all the documents in the n-neighborhood that the label is actually assigned to
        - the total count of appearances of the label in the document
        - the probability that the label translates to the document's title according to the IBM1-translation-model.
        - the count of how often the label directly appears in the title
        
    This algorithm uses the ListNET implementation from [RankLib] in the background. Therefore, it assumes a Java-
    installation to be available and the RankLib-2.5.jar file to be in the same folder as this file.
        
    The whole procedure's main idea was taken from [MLA]. However, as some of their techniques cannot be applied
    in the same way to our application, some of the implementation details differ.
    
    References
    ----------
    [MLA]   Huang, Minlie, Aurélie Névéol, and Zhiyong Lu. 
            "Recommending MeSH terms for annotating biomedical articles." 
            Journal of the American Medical Informatics Association 18.5 (2011): 660-667.
    [RankLib] https://sourceforge.net/p/lemur/wiki/RankLib/
    
    Parameters
    ----------
    n_neighbors: int, default=20
        The size of the neighborhood, from which labels are considered as input to the ranking algorithm
    topk: int, default=5
        The number of labels taken from the top of the list after running the ranking algorithm
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
    """
    def __init__(self, use_lsh_forest=False, n_neighbors=20, topk=5, max_iterations = 300, count_concepts = False, number_of_concepts = 0, count_terms = False, training_validation_split = 0.8, **kwargs ):
        
        self.n_neighbors = n_neighbors
        self.knn = LSHForest(n_neighbors=n_neighbors, **kwargs) if use_lsh_forest else NearestNeighbors(
            n_neighbors=n_neighbors, **kwargs)
        self.y = None
        self.topk = topk
        self.max_iterations = max_iterations
        self.count_concepts = count_concepts
        self.count_terms = count_terms
        self.number_of_concepts = number_of_concepts
        self.training_validation_split = training_validation_split

    def fit(self, X, y):
        self.y = y
        self.knn.fit(X)
        
        distances_to_neighbors,neighbor_id_lists = self.knn.kneighbors()
        
        self._train_translation_model(X,y)
        
        # create features for each label in a documents neighborhood and write them in a file
        self._extract_and_write(X, neighbor_id_lists, distances_to_neighbors, y = y, sum = True)
        # use the RankLib libraries ListNet algorithm to create a ranking model
        subprocess.call(["java", "-jar", "RankLib-2.5.jar", "-train", "listnet_train", "-tvs", str(self.training_validation_split), "-save", "listnet_model", "-ranker" ,"7", "-epoch", str(self.max_iterations)])


    def _train_translation_model(self, X, y):
        translations = []
        print("Creating the translations.")
        
        for row in range(0, X.shape[0]):
            
            title = self._recompose_title(X, row)
            for label in self._get_labels_of_row(row, y):
                translations.append(AlignedSent(title, [label]))
        
    
        #for i,title_string in enumerate(titles):
        #   for label in labels[i]:
        # asent = AlignedSent(title_string.split(),[label])
        # translations.append(asent)
        
        print("Done creating the translations.")
        ibm1 = IBMModel1(translations, 5)
        self.ibm1 = ibm1
    
    def _recompose_title(self, X, row):
        title = []
        for word in X[row].nonzero():
            title.append(str(word))
        
        return title
    
    def _extract_and_write(self, X, neighbor_id_lists, distances_to_neighbors, fileName = "listnet_train", y = None, sum = False):
        listnet_training_samples = open(fileName, "w", encoding='utf-8')
        qid = 1
        
        sum_intersection_percent = 0
        sum_union_percent = 0
        doc_to_neighborhood_dict = {}
        for curr_doc, neighbor_list in enumerate(neighbor_id_lists):
            
            labels_in_neighborhood = self.y[neighbor_list]
            
            labels_per_neighbor = []
            for i in range(0,labels_in_neighborhood.shape[0] - 1):
                cols = self._get_labels_of_row(i,labels_in_neighborhood)
                labels_per_neighbor.append(cols)
                
            all_labels_in_neighborhood = [label for labels in labels_per_neighbor for label in labels]
            all_labels_in_neighborhood = list(set(all_labels_in_neighborhood))
            
            sum_intersection_percent += self._intersection_percent(set(all_labels_in_neighborhood), set(self._get_labels_of_row(0,self.y[curr_doc])))
            sum_union_percent += len(set(all_labels_in_neighborhood)) / len(set(self._get_labels_of_row(0,self.y[curr_doc])))
            
            for label in all_labels_in_neighborhood:
                listnet_training_samples.write(self._generate_line_for_label(label, qid, X, y, curr_doc, labels_per_neighbor, neighbor_list, distances_to_neighbors[curr_doc]))
            
            doc_to_neighborhood_dict[str(qid)] = all_labels_in_neighborhood
            
            qid += 1
        
        if sum:
            self.sum_intersection_percent = 100 * sum_intersection_percent / X.shape[0]
            self.sum_union_percent = 100 * sum_union_percent / X.shape[0]
        return doc_to_neighborhood_dict
                         
    def _intersection_percent(self, all_labels, curr_labels):
        return len(all_labels.intersection(curr_labels)) / len(curr_labels)
    
    def predict(self, X):
        distances, neighbor_id_lists = self.knn.kneighbors(X, n_neighbors=self.n_neighbors)
        doc_to_neighborhood_dict = self._extract_and_write(X, neighbor_id_lists, distances, fileName="listnet_test", y = None)
        
        # use the created ListNET model to get a ranking java -jar RankLib-2.5.jar -rank listnet_test -load listnet_model -ranker 7 -score test_scores
        subprocess.call(["java", "-jar", "RankLib-2.5.jar", "-rank", "listnet_test", "-score", "test_scores", "-load", "listnet_model", "-ranker" ,"7"])
        
        scoresFile = open("test_scores", "r")
        
        results = {}
        for line in scoresFile:
            qid, score = self._extract_score(line)
            if str(qid) in results:
                labels = results[str(qid)]
                labels.append(score)
                results[str(qid)] = labels
            else :
                results[str(qid)] = [score]
            
        predictions = csr_matrix((X.shape[0], self.y.shape[1]))
        
        doc_to_neighborhood_dict = self._extract_topk_score(doc_to_neighborhood_dict, results)
        
        for i in range(0,X.shape[0]):
            for label in doc_to_neighborhood_dict[str(i + 1)]:
                predictions[i, label] = 1
        
        print("In the " + str(self.n_neighbors) + " nearest neighbors there are " + str(self.sum_intersection_percent) + " %% of the labels.")
        print("The number of labels in the " + str(self.n_neighbors) + " nearest neighbors is " + str(self.sum_union_percent))
        return predictions
        
    def _extract_score(self, line):
        entries = line.split()
        qid = entries[0]
        score = entries[2]
        return (qid, score)
    
    def _extract_topk_score(self, doc_to_neighborhood_dict, results):
        for key, value in results.items():
            top_indices = self._get_top_indices(value)
            labels = doc_to_neighborhood_dict[key]
            toplabels = [labels[index] for index in top_indices] 
            doc_to_neighborhood_dict[key] = toplabels
            
        return doc_to_neighborhood_dict
    
    def _get_top_indices(self, array):
        return self._maxEntries(array, self.topk)
    
    # this one was stolen here: http://stackoverflow.com/questions/12787650/finding-the-index-of-n-biggest-elements-in-python-array-list-efficiently
    def _maxEntries(self,a,N):
        return np.argsort(a)[::-1][:N]
    
    def _get_labels_of_row(self,i,matrix):
        labels_of_single_neighbor = matrix[i]
        rows, cols = labels_of_single_neighbor.nonzero()
        return cols
    # we are generating a line per label according to the notation given in http://www.cs.cornell.edu/People/tj/svm_light/svm_rank.html            
    def _generate_line_for_label(self, label, qid, X, y, curr_doc, labels_per_neighbor, neighbor_list, distances_to_neighbor):
        # set the target value to 1 (i.e., the label is ranked high), if the label is actually assigned to the current document, 0 otherwise
        if(y != None): 
            label_in_curr_doc = 1 if label in self._get_labels_of_row(curr_doc, y) else 0
        else:
            label_in_curr_doc = 0
        
        features = self._extract_features(label, curr_doc, labels_per_neighbor, neighbor_list, distances_to_neighbor, X, y)
        featureString = " ".join([str(i + 1) + ":" + str(val) for i,val in enumerate(features)])
        #if y != None:
        line_for_label = str(label_in_curr_doc) + " "
        #else:
                    
        line_for_label = line_for_label + "qid:" + str(qid) + " " + featureString +  "\n"
        return line_for_label
    
    def _extract_features(self, label, curr_doc, labels_per_neighbor, neighbor_list, distances_to_neighbor, X, y):
        sum_similarities = 0
        for i, neighbor_list in enumerate(labels_per_neighbor):
            if label in neighbor_list:
                sum_similarities += distances_to_neighbor[i]
        
        count_in_neighborhood = [some_label for labels in labels_per_neighbor for some_label in labels].count(label)
        
        title = self._recompose_title(X, curr_doc)
        translation_probability = 0
        for word in title:
            translation_probability += self.ibm1.translation_table[word][str(label)]
        
        features = [count_in_neighborhood, sum_similarities, translation_probability]
        
        if self.count_concepts and not self.count_terms:
            features.append(1 if X[curr_doc,label] > 0 else 0)
        elif self.count_concepts and self.count_terms:
            features.append(1 if X[curr_doc, X.shape[1] - self.number_of_concepts + label] > 0 else 0)
        
        return features