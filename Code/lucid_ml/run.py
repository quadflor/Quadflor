#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, json, random, sys, os, argparse, logging, datetime, traceback
from collections import defaultdict
from pprint import pprint
from timeit import default_timer

from classifying.neural_net import MLP, ThresholdingPredictor
from classifying.stack_lin_reg import LinRegStack
from rdflib.plugins.parsers.ntriples import validate

os.environ['OMP_NUM_THREADS'] = '1'  # For parallelization use n_jobs, this gives more control.
import numpy as np
from scipy.stats import entropy
import networkx as nx
import warnings

from utils.processify import processify

from itertools import product

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import KFold, ShuffleSplit
# from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, make_pipeline, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
import scipy.sparse as sps

# imports for hyperparameter optimization
from bayes_opt import BayesianOptimization
from hyperopt import fmin, rand, hp
from sklearn.gaussian_process.kernels import Matern

from classifying.br_kneighbor_classifier import BRKNeighborsClassifier
from classifying.kneighbour_l2r_classifier import KNeighborsL2RClassifier
from classifying.meancut_kneighbor_classifier import MeanCutKNeighborsClassifier
from classifying.nearest_neighbor import NearestNeighbor
from classifying.rocchioclassifier import RocchioClassifier
from classifying.stacked_classifier import ClassifierStack
from classifying.tensorflow_models import MultiLabelSKFlow, mlp_base, mlp_soph, cnn, lstm
from utils.Extractor import load_dataset
from utils.metrics import hierarchical_f_measure, f1_per_sample
from utils.nltk_normalization import NltkNormalizer, word_regexp, character_regexp
from utils.persister import Persister
from weighting.SpreadingActivation import SpreadingActivation, BinarySA, OneHopActivation
from weighting.synset_analysis import SynsetAnalyzer
from weighting.bm25transformer import BM25Transformer
from weighting.concept_analysis import ConceptAnalyzer
from weighting.graph_score_vectorizer import GraphVectorizer
from utils.text_encoding import TextEncoder

### SET LOGGING
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def _build_features(options):
    DATA_PATHS = json.load(options.key_file)
    VERBOSE = options.verbose
    persister = Persister(DATA_PATHS, options)
    if options.persist and persister.is_saved():
        X, Y, tr = persister.read()
        if VERBOSE: print("Y = " + str(Y.shape))
    else:
        # --- LOAD DATA ---
        fold_list = None
        if options.fixed_folds:
            X_raw, Y_raw, tr, fold_list = load_dataset(DATA_PATHS, options.data_key, options.fulltext, fixed_folds=True)
        else:
            X_raw, Y_raw, tr, _ = load_dataset(DATA_PATHS, options.data_key, options.fulltext, fixed_folds=False)
        if options.toy_size < 1:
            if VERBOSE: print("Just toying with %d%% of the data." % (options.toy_size * 100))
            zipped = list(zip(X_raw, Y_raw))
            random.shuffle(zipped)
            X_raw, Y_raw = zip(*zipped)
            toy_slice = int(options.toy_size * len(X_raw))
            X_raw, Y_raw = X_raw[:toy_slice], Y_raw[:toy_slice]

        if options.verbose: print("Binarizing labels...")
        mlb = MultiLabelBinarizer(sparse_output=True, classes=[i[1] for i in sorted(
            tr.index_nodename.items())] if options.hierarch_f1 else None)
        Y = mlb.fit_transform(Y_raw)
        
        if VERBOSE: 
            print("Y = " + str(Y.shape))
            y_sum = Y.sum(axis = 0)
            for i in range(1, 5):
                print("Number of labels assigned more than", i, "times:" , np.sum(y_sum > i))

        # --- EXTRACT FEATURES ---
        input_format = 'filename' if options.fulltext else 'content'
        if options.concepts:
            if tr is None:
                raise ValueError("Unable to extract concepts, since no thesaurus is given!")
            concept_analyzer = SynsetAnalyzer().analyze if options.synsets \
                else ConceptAnalyzer(tr.thesaurus, input=input_format, persist=options.persist and options.concepts,
                    persist_dir=options.persist_to, repersist=options.repersist,
                    file_path=DATA_PATHS[options.data_key]['X']).analyze
            concepts = CountVectorizer(input=input_format, analyzer=concept_analyzer, binary=options.binary,
                                       vocabulary=tr.nodename_index if not options.synsets else None,
                                       ngram_range=(1, options.ngram_limit))
            
        # pick max_features from each ngram-group
        if options.max_features is not None and options.ngram_limit > 1:
            ngram_vectorizers = []
            for i in range(1, options.ngram_limit + 1):
                i_grams = CountVectorizer(input=input_format, stop_words='english', binary=options.binary,
                                    token_pattern=word_regexp, max_features=options.max_features, 
                                    ngram_range=(i, i))
                ngram_vectorizers.append(i_grams)
            terms = FeatureUnion([(str(i) + "_gram", t) for i, t in enumerate(ngram_vectorizers)])
        else:
            terms = CountVectorizer(input=input_format, stop_words='english', binary=options.binary,
                                    token_pattern=word_regexp, max_features=options.max_features, 
                                    ngram_range=(1, options.ngram_limit))
        if options.charngrams:
            character_ngrams = CountVectorizer(input=input_format, binary=options.binary,
                                    token_pattern=character_regexp, max_features=options.max_features, 
                                    ngram_range=(1, options.char_ngram_limit),
                                    analyzer = 'char_wb')

        if options.hierarchical:
            hierarchy = tr.nx_graph
            if options.prune_tree:
                if VERBOSE: print("[Pruning] Asserting tree hierarchy...")
                old_edge_count = hierarchy.number_of_edges()
                hierarchy = nx.bfs_tree(hierarchy, tr.nx_root)
                pruned = old_edge_count - hierarchy.number_of_edges()
                if VERBOSE: print("[Pruning] Pruned %d of %d edges (%.2f) to assert a tree hierarchy" % (pruned, old_edge_count, pruned/old_edge_count))

            if options.hierarchical == "bell":
                activation = SpreadingActivation(hierarchy, decay=1, weighting="bell", root=tr.nx_root)
            elif options.hierarchical == "belllog":
                activation = SpreadingActivation(hierarchy, decay=1, weighting="belllog", root=tr.nx_root)
            elif options.hierarchical == "children":
                # weights are already initialized with 1/out_degree, so use basic SA with decay 1
                activation = SpreadingActivation(hierarchy, decay=1, weighting="children")
            elif options.hierarchical == "binary":
                activation = BinarySA(hierarchy)
            elif options.hierarchical == "onehop":
                activation = OneHopActivation(hierarchy, verbose=VERBOSE)
            else:
                #  basic
                activation = SpreadingActivation(tr.nx_graph, firing_threshold=1.0, decay=0.25, weighting=None)
            concepts = make_pipeline(concepts, activation)

        features = []
        if options.graph_scoring_method:
            features.append(("graph_vectorizer", GraphVectorizer(method=options.graph_scoring_method, analyzer=concept_analyzer
            if options.concepts else NltkNormalizer().split_and_normalize)))
        if options.terms:
            features.append(("terms", terms))
        if options.concepts:
            features.append(("concets", concepts))
        if options.charngrams:
            features.append(("char_ngrams", character_ngrams))
        if options.onehot:
            features.append(("onehot", TextEncoder(input_format = "filename" if options.fulltext else "content", 
                                    max_words=options.max_features, pretrained = options.pretrained_embeddings)))
        if len(features) == 0:
            raise ValueError("No feature representation specified!")
        
        extractor = FeatureUnion(features)

        if VERBOSE: print("Extracting features...")
        if VERBOSE > 1: start_ef = default_timer()
        X = extractor.fit_transform(X_raw)
        if VERBOSE > 1: print(default_timer() - start_ef)
        if options.persist:
            persister.persist(X, Y, tr)
            
    return X, Y, extractor, mlb, fold_list, X_raw, Y_raw, tr

def _print_feature_info(X, options):
    VERBOSE = options.verbose
    if VERBOSE:
        print("Feature size: {}".format(X.shape[1]))
        print("Number of documents: {}".format(X.shape[0]))
        # these printouts only make sense if we have BoW representation 
        if sps.issparse(X):
            print("Mean distinct words per document: {}".format(X.count_nonzero() /
                                                        X.shape[0]))
            words = X.sum(axis=1)
            print("Mean word count per document: {} ({})".format(words.mean(), words.std()))

    if VERBOSE > 1:
        X_tmp = X.todense()
        # drop samples without any features...
        X_tmp = X_tmp[np.unique(np.nonzero(X_tmp)[0])]
        print("[entropy] Dropped {} samples with all zeroes?!".format(X.shape[0] - X_tmp.shape[0]))
        X_tmp = X_tmp.T # transpose to compute entropy per sample
        h = entropy(X_tmp)
        print("[entropy] shape:", h.shape)
        print("[entropy] mean entropy per sample {} ({})".format(h.mean(), h.std()))
        # print("Mean entropy (base {}): {}".format(X_dense.shape[0], entropy(X_dense, base=X_dense.shape[0]).mean()))
        # print("Mean entropy (base e): {}".format(entropy(X_dense).mean()))
    # _, _, values = sp.find(X)
    # print("Mean value: %.2f (+/- %.2f) " % (values.mean(), 2 * values.std()))


    # n_iter = np.ceil(10**6 / (X.shape[0] * 0.9))
    # print("Dynamic n_iter = %d" % n_iter)

def _check_interactive(options, X, Y, extractor, mlb):
    if options.interactive:
        print("Please wait...")
        clf = create_classifier(options, Y.shape[1])  # --- INTERACTIVE MODE ---
        clf.fit(X, Y)
        thesaurus = tr.thesaurus
        print("Ready.")
        try:
            for line in sys.stdin:
                x = extractor.transform([line])
                y = clf.predict(x)
                desc_ids = mlb.inverse_transform(y)[0]
                labels = [thesaurus[desc_id]['prefLabel'] for desc_id in desc_ids]
                print(*labels)
        except KeyboardInterrupt:
            exit(1)
        exit(0)

def _build_folds(options, fold_list):
    validation_set_indices = None
    
    if options.cross_validation:
        kf = KFold(n_splits=options.folds, shuffle=True)
    elif options.fixed_folds:
        fixed_folds = []

        # TODO: we assume 10 normal folds and 1 folds with extra samples. need to generalize
        basic_folds = range(10)
        
        # we assume the extra data to be in the last fold
        # TODO: currently we assume 10 folds (+1 extra)
        extra_data = [index for index,x in enumerate(fold_list) if x == 10]
        
        validation_set_indices = []
        for i in range(options.folds):
            
            training_fold = [index for index,x in enumerate(fold_list) if x in basic_folds and x != i]
            
            if options.validation_size > 0:
                # separate validation from training set here, and rejoin later if appropriate
                num_validation_samples = int(len(training_fold) * options.validation_size)
                validation_set_indices.append(training_fold[:num_validation_samples])
                training_fold = training_fold[num_validation_samples:]
            
            # add more training data from extra samples
            if options.extra_samples_factor > 1:
                num_extra_samples = int(min((options.extra_samples_factor - 1) * len(training_fold), len(extra_data)))
                training_fold += extra_data[:num_extra_samples]
            
            test_fold = [index for index,x in enumerate(fold_list) if x == i]
            fixed_folds.append((training_fold, test_fold))
          
        # helper class to conform sklearn's model_selection structure
        class FixedFoldsGenerator():
            def split(self, X):
                return fixed_folds
        
        kf = FixedFoldsGenerator()
            
    else:
        kf = ShuffleSplit(test_size=options.test_size, n_splits = 1)
        
    return kf, validation_set_indices

def _run_experiment(X, Y, kf, validation_set_indices, mlb, X_raw, Y_raw, tr, options):
    VERBOSE = options.verbose
    scores = defaultdict(list)
    if options.plot:
        all_f1s = []
    
    for iteration, (train, test) in enumerate(kf.split(X)):
        
        if VERBOSE: print("=" * 80)
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        
        clf = create_classifier(options, Y_train.shape[1])  # --- INTERACTIVE MODE ---
        
        # extract a validation set and inform the classifier where to find it
        if options.validation_size > 0:
            # if we don't have fixed folds, we may pick the validation set randomly
            if options.cross_validation or options.one_fold:
                train, val = next(ShuffleSplit(test_size=options.validation_size, n_splits = 1).split(X_train))
                X_train, X_val, Y_train, Y_val = X_train[train], X_train[val], Y_train[train], Y_train[val]
            elif options.fixed_folds:
                X_train = X_train
                X_val = X[validation_set_indices[iteration]]
                Y_val = Y[validation_set_indices[iteration]]
                
            # put validation data at the end of training data and tell classifier the position where they start, if it is able
            _, estimator = clf.steps[-1]
            if hasattr(estimator, 'validation_data_position'):
                estimator.validation_data_position = X_train.shape[0] 
            else:
                raise ValueError("Validation size given although the estimator has no 'validation_data_position' property!")
            
            if sps.issparse(X):
                X_train = sps.vstack((X_train, X_val))
            else:
                X_train = np.vstack((X_train, X_val))
                
            if sps.issparse(Y):
                Y_train = sps.vstack((Y_train, Y_val))
            else:
                Y_train = np.vstack((Y_train, Y_val))

        # mlp doesn't seem to like being stuck into a new process...
        if options.debug or options.clf_key in {'mlp', 'mlpthr', 'mlpsoph', "cnn", "mlpbase", "lstm"}:
            Y_pred, Y_train_pred = fit_predict(X_test, X_train, Y_train, options, tr, clf)
        else:
            Y_pred, Y_train_pred = fit_predict_new_process(X_test, X_train, Y_train, options, tr, clf)

        if options.training_error:
            scores['train_f1_samples'].append(f1_score(Y_train, Y_train_pred, average='samples'))

        scores['avg_n_labels_pred'].append(np.mean(Y_pred.getnnz(1)))
        scores['avg_n_labels_gold'].append(np.mean(Y_test.getnnz(1)))
        scores['f1_samples'].append(f1_score(Y_test, Y_pred, average='samples'))
        scores['p_samples'].append(precision_score(Y_test, Y_pred, average='samples'))
        scores['r_samples'].append(recall_score(Y_test, Y_pred, average='samples'))
        scores['f1_micro'].append(f1_score(Y_test, Y_pred, average='micro'))
        scores['p_micro'].append(precision_score(Y_test, Y_pred, average='micro'))
        scores['r_micro'].append(recall_score(Y_test, Y_pred, average='micro'))
        scores['f1_macro'].append(f1_score(Y_test, Y_pred, average='macro'))
        scores['p_macro'].append(precision_score(Y_test, Y_pred, average='macro'))
        scores['r_macro'].append(recall_score(Y_test, Y_pred, average='macro'))
        if options.plot:
            all_f1s.append(f1_per_sample(Y_test, Y_pred))

        if options.worst:
            f1s = f1_per_sample(Y_test, Y_pred)
            predicted_labels = [[tr.thesaurus[l]['prefLabel'] for l in y] for y in mlb.inverse_transform(Y_pred)]
            f1s_ids = sorted(zip(f1s, [X_raw[i] for i in test],
                                 [[tr.thesaurus[l]['prefLabel'] for l in Y_raw[i]] for i in test], predicted_labels))
            pprint(f1s_ids[:options.worst])

        if options.hierarch_f1:
            scores['hierarchical_f_score'].append(
                hierarchical_f_measure(tr, Y_test, Y_pred))

        if options.cross_validation and VERBOSE:
            print(' <> '.join(["%s : %0.3f" % (key, values[-1]) for key, values in sorted(scores.items())]))
            # if options.lsa:
            #     if VERBOSE: print("Variance explained by SVD:", svd.explained_variance_ratio_.sum())

    if VERBOSE: print("=" * 80)

    results = {key: (np.array(values).mean(), np.array(values).std()) for key, values in scores.items()}

    print(' <> '.join(["%s: %0.3f (+/- %0.3f)" % (key, mean, std) for key, (mean, std) in sorted(results.items())]))

    if options.output_file:
        write_to_csv(results, options)

    if options.plot:
        Y_f1 = np.hstack(all_f1s)
        Y_f1.sort()
        if VERBOSE:
            print("Y_f1.shape:", Y_f1.shape, file=sys.stderr)
            print("Saving f1 per document as txt numpy to", options.plot)
        np.savetxt(options.plot, Y_f1)

    return results

def _update_options(options, **parameters):
    for param_name, param_value in parameters.items():
        print("In automatic optimization trying parameter:", param_name, "with value", param_value)
        
        try:
            setattr(options, param_name, param_value)
        except AttributeError:
            print("Can't find parameter ", param_name, "so we'll not use it.")
            continue
        
    return options


def _make_space(options):
    
    space = {}
    inits = {}
    with open(options.optimization_spaces) as optimization_file:
        for line in optimization_file:
            
            # escape comments
            if line.startswith("#"):
                continue
            
            line = line.strip()
            info = line.split(",")
            param_name = info[0]
            
            if options.optimization == "random":
                left_bound, right_bound = float(info[1]), float(info[2])
                
                param_type = info[3]
                
                try:
                    param_type = getattr(hp, param_type)
                except AttributeError:
                    print("hyperopt has no attribute", param_type)
                    continue
                
                space[param_name] = param_type(param_name, left_bound, right_bound)  
            elif options.optimization == "bayesian":
                left_bound, right_bound = float(info[1]), float(info[2])
                
                init_values = list(map(float, info[3:]))
                num_init_vals = len(init_values)
                inits[param_name] = init_values
                space[param_name] = (left_bound, right_bound)
                
            elif options.optimization == "grid":
                
                param_type = info[1]
                if param_type == "int":
                    cast_func = int
                elif param_type == "float":
                    cast_func = float
                elif param_type == "string":
                    cast_func = str
                
                # all possible values
                space[param_name] = list(map(cast_func, info[2:]))
    
    if options.optimization == "bayesian":
        return space, inits, num_init_vals
    else:
        return space
    
def _all_option_combinations(space):
    
    names = [name for name, _ in space.items()]
    values = [values for _, values in space.items()]
    
    val_combinations = product(*values)
    
    combinations = []
    for combi in val_combinations:
        new_param_dict = {}
        for i, val in enumerate(combi):
            new_param_dict[names[i]] = val
        
        combinations.append(new_param_dict)
    
    return combinations
    
def run(options):
    VERBOSE = options.verbose
    
    ### SET SEEDS FOR REPRODUCABILITY
    np.random.seed(1337)
    random.seed(1337)
    ###
    
    # load dataset and build feature representation
    X, Y, extractor, mlb, fold_list, X_raw, Y_raw, tr = _build_features(options)
    _print_feature_info(X, options)

    # go to interactive mode if on
    _check_interactive(options, X, Y, extractor, mlb)

    if VERBOSE: print("Performing %d-fold cross-validation..." % (options.folds if options.cross_validation else 1))

    # prepare validation over folds
    kf, validation_set_indices = _build_folds(options, fold_list)
    
    if options.optimization:
        
        def optimized_experiment(**parameters):
            
            current_options = _update_options(options, **parameters)
            results = _run_experiment(X, Y, kf, validation_set_indices, mlb, X_raw, Y_raw, tr, current_options)
        
            # return the f1 score of the previous experiment
            return results["f1_samples"][0]
        
        if options.optimization == "bayesian":
            
            gp_params = {"alpha": 1e-5, "kernel" : Matern(nu = 5 / 2)}
            space, init_vals, num_init_vals = _make_space(options)
            bayesian_optimizer = BayesianOptimization(optimized_experiment, space)
            bayesian_optimizer.explore(init_vals)
            bayesian_optimizer.maximize(n_iter=options.optimization_iterations - num_init_vals,
                                        acq = 'ei',
                                        **gp_params)
            
        elif options.optimization == "random":
            
            fmin(lambda parameters : optimized_experiment(**parameters),
                        _make_space(options),
                        algo=rand.suggest,
                        max_evals=options.optimization_iterations,
                        rstate = np.random.RandomState(1337))
            
        elif options.optimization == "grid":
            # perform grid-search by running every possible parameter combination
            combinations = _all_option_combinations(_make_space(options))
            for combi in combinations:
                optimized_experiment(**combi)
            
    else:
        results = _run_experiment(X, Y, kf, validation_set_indices, mlb, X_raw, Y_raw, tr, options)

def fit_predict(X_test, X_train, Y_train, options, tr, clf):
    if options.verbose: print("Fitting", X_train.shape[0], "samples...")
    clf.fit(X_train, Y_train)

    if options.training_error:
        if options.verbose: print("Predicting", X_train.shape[0], "training samples...")
        Y_pred_train = clf.predict(X_train)
    else:
        Y_pred_train = None

    if options.verbose: print("Predicting", X_test.shape[0], "samples...")
    Y_pred = clf.predict(X_test)
    return Y_pred, Y_pred_train

@processify
def fit_predict_new_process(X_test, X_train, Y_train, options, tr, clf):
    return fit_predict(X_test, X_train, Y_train, options, tr, clf)

def create_classifier(options, num_concepts):
    # Learning 2 Rank algorithm name to ranklib identifier mapping
    l2r_algorithm = {'listnet' : "7",
                     'adarank' : "3",
                     'ca' : "4",
                     'lambdamart' : "6"}

    # --- BUILD CLASSIFIER ---
    sgd = OneVsRestClassifier(SGDClassifier(loss='log', n_iter=options.max_iterations, verbose=max(0,options.verbose-2), penalty=options.penalty, alpha=options.alpha, average=True),
        n_jobs=options.jobs)
    logregress = OneVsRestClassifier(LogisticRegression(C=64, penalty='l2', dual=False, verbose=max(0,options.verbose-2)),
        n_jobs=options.jobs)
    l2r_classifier = KNeighborsL2RClassifier(n_neighbors=options.l2r_neighbors, max_iterations=options.max_iterations,
                                               count_concepts=True if options.concepts else False,
                                               number_of_concepts=num_concepts,
                                               count_terms=True if options.terms else False,
                                               algorithm='brute', metric='cosine',
                                               algorithm_id = l2r_algorithm[options.l2r],
                                               l2r_metric = options.l2r_metric + "@20",
                                               n_jobs = options.jobs,
                                               translation_probability = options.translation_prob)
    mlp = MLP(verbose=options.verbose, batch_size = options.batch_size, learning_rate = options.learning_rate, epochs = options.max_iterations)
    classifiers = {
        "nn": NearestNeighbor(use_lsh_forest=options.lshf),
        "brknna": BRKNeighborsClassifier(mode='a', n_neighbors=options.k, use_lsh_forest=options.lshf,
                                         algorithm='brute', metric='cosine', auto_optimize_k=options.grid_search),
        "brknnb": BRKNeighborsClassifier(mode='b', n_neighbors=options.k, use_lsh_forest=options.lshf,
                                         algorithm='brute', metric='cosine', auto_optimize_k=options.grid_search),
        "listnet": l2r_classifier,
        "l2rdt": ClassifierStack(base_classifier=l2r_classifier, n_jobs=options.jobs, n=options.k, dependencies=options.label_dependencies),
        "mcknn": MeanCutKNeighborsClassifier(n_neighbors=options.k, algorithm='brute', metric='cosine', soft=False),
        # alpha 10e-5
        "bbayes": OneVsRestClassifier(BernoulliNB(alpha=options.alpha), n_jobs=options.jobs),
        "mbayes": OneVsRestClassifier(MultinomialNB(alpha=options.alpha), n_jobs=options.jobs),
        "lsvc": OneVsRestClassifier(LinearSVC(C=4, loss='squared_hinge', penalty='l2', dual=False, tol=1e-4),
                                    n_jobs=options.jobs),
        "logregress": logregress,
        "sgd": sgd,
        "rocchio": RocchioClassifier(metric = 'cosine', k = options.k),
        "sgddt": ClassifierStack(base_classifier=sgd, n_jobs=options.jobs, n=options.k),
        "rocchiodt": ClassifierStack(base_classifier=RocchioClassifier(metric = 'cosine'), n_jobs=options.jobs, n=options.k),
        "logregressdt": ClassifierStack(base_classifier=logregress, n_jobs=options.jobs, n=options.k),
        "mlp": mlp,
        "mlpbase" : MultiLabelSKFlow(batch_size = options.batch_size,
                                     num_epochs=options.max_iterations,
                                     learning_rate = options.learning_rate,
                                     tf_model_path = options.tf_model_path,
                                     optimize_threshold = options.optimize_threshold,
                                     get_model = mlp_base(hidden_activation_function = options.hidden_activation_function),
                                     patience = options.patience,
                                     num_steps_before_validation = options.num_steps_before_validation,
                                     bottleneck_layers = options.bottleneck_layers,
                                     hidden_keep_prob = options.dropout,
                                     gpu_memory_fraction = options.memory),
        "mlpsoph" : MultiLabelSKFlow(batch_size = options.batch_size,
                                     num_epochs=options.max_iterations,
                                     learning_rate = options.learning_rate,
                                     tf_model_path = options.tf_model_path,
                                     optimize_threshold = options.optimize_threshold,
                                     get_model = mlp_soph(options.dropout, options.embedding_size,
                                                          hidden_layers = options.hidden_layers, self_normalizing = options.snn,
                                                          standard_normal = options.standard_normal,
                                                          batch_norm = options.batch_norm,
                                                          hidden_activation_function = options.hidden_activation_function
                                                          ),
                                     patience = options.patience,
                                     num_steps_before_validation = options.num_steps_before_validation,
                                     bottleneck_layers = options.bottleneck_layers,
                                     hidden_keep_prob = options.dropout,
                                     gpu_memory_fraction = options.memory,
                                     meta_labeler_phi = options.meta_labeler_phi,
                                     meta_labeler_alpha = options.meta_labeler_alpha,
                                     meta_labeler_min_labels = options.meta_labeler_min_labels,
                                     meta_labeler_max_labels = options.meta_labeler_max_labels),
        "cnn": MultiLabelSKFlow(batch_size = options.batch_size,
                                     num_epochs=options.max_iterations,
                                     learning_rate = options.learning_rate,
                                     tf_model_path = options.tf_model_path,
                                     optimize_threshold = options.optimize_threshold,
                                     patience = options.patience,
                                     num_steps_before_validation = options.num_steps_before_validation,
                                     get_model = cnn(options.dropout, options.embedding_size, 
                                                          hidden_layers = options.hidden_layers, 
                                                          pretrained_embeddings_path = options.pretrained_embeddings,
                                                          trainable_embeddings=options.trainable_embeddings,
                                                          dynamic_max_pooling_p=options.dynamic_max_pooling_p),
                                     bottleneck_layers = options.bottleneck_layers,
                                     hidden_keep_prob = options.dropout,
                                     gpu_memory_fraction = options.memory,
                                     meta_labeler_phi = options.meta_labeler_phi,
                                     meta_labeler_alpha = options.meta_labeler_alpha,
                                     meta_labeler_min_labels = options.meta_labeler_min_labels,
                                     meta_labeler_max_labels = options.meta_labeler_max_labels),
        "lstm": MultiLabelSKFlow(batch_size = options.batch_size,
                                     num_epochs=options.max_iterations,
                                     learning_rate = options.learning_rate,
                                     tf_model_path = options.tf_model_path,
                                     optimize_threshold = options.optimize_threshold,
                                     patience = options.patience,
                                     num_steps_before_validation = options.num_steps_before_validation,
                                     get_model = lstm(options.dropout, options.embedding_size, 
                                                          hidden_layers = options.hidden_layers, 
                                                          pretrained_embeddings_path = options.pretrained_embeddings,
                                                          trainable_embeddings = options.trainable_embeddings,
                                                          variational_recurrent_dropout = options.variational_recurrent_dropout,
                                                          bidirectional = options.bidirectional,
                                                          aggregate_output = options.aggregate_output,
                                                          iterate_until_maxlength = options.iterate_until_maxlength),
                                     bottleneck_layers = options.bottleneck_layers,
                                     hidden_keep_prob = options.dropout,
                                     gpu_memory_fraction = options.memory,
                                     meta_labeler_phi = options.meta_labeler_phi,
                                     meta_labeler_alpha = options.meta_labeler_alpha,
                                     meta_labeler_min_labels = options.meta_labeler_min_labels,
                                     meta_labeler_max_labels = options.meta_labeler_max_labels),
        "nam": ThresholdingPredictor(MLP(verbose=options.verbose, final_activation='sigmoid', batch_size = options.batch_size, 
                                         learning_rate = options.learning_rate, 
                                         epochs = options.max_iterations), 
                                     alpha=options.alpha, stepsize=0.01, verbose=options.verbose),
        "mlpthr": LinRegStack(mlp, verbose=options.verbose),
        "mlpdt" : ClassifierStack(base_classifier=mlp, n_jobs=options.jobs, n=options.k)
    }
    # Transformation: either bm25 or tfidf included in pipeline so that IDF of test data is not considered in training
    norm = "l2" if options.norm else None
    if options.bm25:
        trf = BM25Transformer(sublinear_tf=True if options.lsa else False, use_idf=options.idf, norm=norm,
                              bm25_tf=True, use_bm25idf=True)
    elif options.terms or options.concepts:
        trf = TfidfTransformer(sublinear_tf=True if options.lsa else False, use_idf=options.idf, norm=norm)

    # Pipeline with final estimator ##
    if options.graph_scoring_method or options.clf_key in ["bbayes", "mbayes"]:
        clf = classifiers[options.clf_key]

    # elif options.lsa:
    #     svd = TruncatedSVD(n_components=options.lsa)
    #     lsa = make_pipeline(svd, Normalizer(copy=False))
    #     clf = Pipeline([("trf", trf), ("lsa", lsa), ("clf", classifiers[options.clf_key])])
    elif options.terms or options.concepts:
        clf = Pipeline([("trf", trf), ("clf", classifiers[options.clf_key])])
    else:
        clf = Pipeline([("clf", classifiers[options.clf_key])])
    return clf

def _generate_parsers():
    # meta parser to handle config files
    meta_parser = argparse.ArgumentParser(add_help=False)
    meta_parser.add_argument('-C', '--config-file', dest='config_file', type=argparse.FileType('r'), default=None, help= \
    "Specify a config file containing lines of execution arguments")
    meta_parser.add_argument('-d', '--dry', dest='dry', action='store_true', default=False, help= \
            "Do nothing but validate command line and config file parameters")

    ### Parser for the usual command line arguments
    parser = argparse.ArgumentParser(parents=[meta_parser])
    parser.add_argument('-j', type=int, dest='jobs', default=1, help="Number of jobs (processes) to use when something can be parallelized. -1 means as many as possible.")
    parser.add_argument('-o', '--output', dest="output_file", type=str, default='', help= \
        "Specify the file name to save the result in. Default: [None]")
    parser.add_argument('-O',
                    '--plot',
                    type=str,
                    default=None,
                    help='Plot results to FNAME',
                    metavar='FNAME')
    parser.add_argument('-v', '--verbose', default=0, action="count", help=\
            "Specify verbosity level -v for 1, -vv for 2, ... [0]")
    parser.add_argument('--debug', action="store_true", dest="debug", default=False, help=
    "Enables debug mode. Makes fit_predict method debuggable by not starting a single fold in a new process.")

    metric_options = parser.add_argument_group()
    metric_options.add_argument('-r', action='store_true', dest='hierarch_f1', default=False, help=
    'Calculate hierarchical f-measure (Only usable')
    metric_options.add_argument('--worst', type=int, dest='worst', default=0, help=
    'Output given number of top badly performing samples by f1_measure.')

    # mutually exclusive group for executing
    execution_options = parser.add_mutually_exclusive_group(required=True)
    execution_options.add_argument('-x', action="store_true", dest="one_fold", default=False, help=
    "Run on one fold [False]")
    execution_options.add_argument('-X', action="store_true", dest="cross_validation", default=False, help=
    "Perform cross validation [False]")
    execution_options.add_argument('--fixed_folds', action="store_true", dest="fixed_folds", default=False, help=
    "Perform cross validation with fixed folds.")
    execution_options.add_argument('-i', '--interactive', action="store_true", dest="interactive", default=False, help= \
        "Use whole supplied data as training set and classify new inputs from STDIN")


    # be a little versatile
    detailed_options = parser.add_argument_group("Detailed Execution Options")
    detailed_options.add_argument('--test-size', type=float, dest='test_size', default=0.1, help=
    "Desired relative size for the test set [0.1]")
    detailed_options.add_argument('--optimization', type=str, dest='optimization', default=None, help=
    "Whether to use Random Search or Bayesian Optimization for hyperparameter search. [None]", choices = ["grid", "random", "bayesian", None])
    detailed_options.add_argument('--optimization_spaces', type=str, dest='optimization_spaces', default="default_searchspace", help=
    "Path to a file that specifies the search spaces for hyperparameters [default_searchspace]")
    detailed_options.add_argument('--optimization_iterations', type=int, dest='optimization_iterations', default=10, help=
    "Number of iterations in hyperparameter search. [10]")
    detailed_options.add_argument('--val-size', type=float, dest='validation_size', default=0., help=
    "Desired relative size of the training set used as validation set [0.]")
    detailed_options.add_argument('--folds', type=int, dest='folds', default=10, help=
    "Number of folds used for cross validation [10]")
    detailed_options.add_argument('--toy', type=float, dest='toy_size', default=1.0, help=
    "Eventually use a smaller block of the data set from the very beginning. [1.0]")
    detailed_options.add_argument('--extra_samples_factor', type=float, dest='extra_samples_factor', default=1.0, help=
    "This option only has an effect when the '--fixed_folds' option is true. The value determines the factor 'x >= 1' by which\
    the training set is enriched with samples from the 11th fold. Hence, the total number of training data will be \
    x * size of tranining set. By default, the value is x = 1.")
    detailed_options.add_argument('--training-error', action="store_true", dest="training_error", default=False, help=\
            "Compute training error")

    # options to specify the dataset
    data_options = parser.add_argument_group("Dataset Options")
    data_options.add_argument('-F', '--fulltext', action="store_true", dest="fulltext", default=False,
                              help="Fulltext instead of titles")
    data_options.add_argument('-k', '--key-file', dest="key_file", type=argparse.FileType('r'), default="file_paths.json", help=\
                                  "Specify the file to use as Key file for -K")
    data_options.add_argument('-K', '--datakey', type=str, dest="data_key", default='example-titles', help="Prestored key of data.")


    # group for feature_options
    feature_options = parser.add_argument_group("Feature Options")
    feature_options.add_argument('-c', '--concepts', action="store_true", dest="concepts", default=False, help= \
        "use concepts [False]")
    feature_options.add_argument('-t', '--terms', action="store_true", dest="terms", default=False, help= \
        "use terms [True]")
    feature_options.add_argument('--charngrams', action="store_true", dest="charngrams", default=False, help= \
        "use character n-grams [True]")
    feature_options.add_argument('--onehot', action="store_true", dest="onehot", default=False, help= \
        "Encode the input words as one hot. [True]")
    feature_options.add_argument('--max_features', type=int, dest="max_features", default=None, help= \
        "Specify the maximal number of features to be considered for a BoW model [None, i.e., infinity]")
    feature_options.add_argument('-s', '--synsets', action="store_true", dest="synsets", default=False, help= \
        "use synsets [False]")
    feature_options.add_argument('-g', '--graphscoring', dest="graph_scoring_method", type=str, default="", \
                                 help="Use graphscoring method instead of concepts and/or terms", \
                                 choices=["degree", "betweenness", "pagerank", "hits", "closeness", "katz"])
    feature_options.add_argument('--prune', action="store_true", dest="prune_tree", default=False, help="Prune polyhierarchy to tree")
    feature_options.add_argument('-H', type=str, dest="hierarchical", default="", \
                                 help="Perform spreading activation.", \
                                 choices=['basic', 'bell', 'belllog', 'children', 'binary', 'onehop'])
    feature_options.add_argument('-B', '--bm25', dest="bm25", action="store_true", default=False, help=
    "Use BM25 instead of TFIDF for final feature transformation")
    feature_options.add_argument('-b', '--binary', action="store_true", dest="binary", default=False, help=
    "do not count the words but only store their prevalence in a document")
    feature_options.add_argument('--no-idf', action="store_false", dest="idf", default=True, help=
    "Do not use IDF")
    feature_options.add_argument('--no-norm', action="store_false", dest="norm",
                                 default=True, help="Do not normalize values")
    feature_options.add_argument('--ngram_limit', type=int, dest="ngram_limit", default=1, help= \
        "Specify the n for n-grams to take into account for token-based BoW vectorization. [1]")
    feature_options.add_argument('--char_ngram_limit', type=int, dest="char_ngram_limit", default=3, help= \
        "Specify the n for character n-grams to take into account for character n-gram based BoW vectorization. [3]")

    # group for classifiers
    classifier_options = parser.add_argument_group("Classifier Options")
    classifier_options.add_argument('-f', '--classifier', dest="clf_key", default="nn", help=
    "Specify the final classifier.", choices=["nn", "brknna", "brknnb", "bbayes", "mbayes", "lsvc",
                                              "sgd", "sgddt", "rocchio", "rocchiodt", "logregress", "logregressdt",
                                              "mlp", "listnet", "l2rdt", 'mlpthr', 'mlpdt', 'nam', 'mlpbase', "mlpsoph", "cnn", "lstm"])
    classifier_options.add_argument('-a', '--alpha', dest="alpha", type=float, default=1e-7, help= \
        "Specify alpha parameter for stochastic gradient descent")
    classifier_options.add_argument('-n', dest="k", type=int, default=1, help=
    "Specify k for knn-based classifiers. Also used as the count of meta-classifiers considered for each sample in multi value stacking approaches [1]")
    classifier_options.add_argument('-l', '--lshf', action="store_true", dest="lshf", default=False, help=
    "Approximate nearest neighbors using locality sensitive hashing forests")
    classifier_options.add_argument('-L', '--LSA', type=int, dest="lsa", default=None, help=
    "Use Latent Semantic Analysis / Truncated Singular Value Decomposition\
            with n_components output dimensions", metavar="n_components")
    classifier_options.add_argument('-G', '--grid-search', action="store_true", dest="grid_search", default=False, help=
    "Performs Grid search to find optimal K")
    classifier_options.add_argument('-e', type=int, dest="max_iterations", default=5, help=
    "Determine the number of epochs for the training of several classifiers [5]")
    classifier_options.add_argument('--patience', type=int, dest="patience", default=5, help=
    "Specify the number of steps of no improvement in validation score before training is stopped. [5]")
    classifier_options.add_argument('--num_steps_before_validation', type=int, dest="num_steps_before_validation", default=None, help=
    "Specify the number of steps before evaluating on the validation set. [None]")
    classifier_options.add_argument('--learning_rate', type=float, dest="learning_rate", default=None, help=
    "Determine the learning rate for training of several classifiers. If set to 'None', the learning rate is automatically based on an empirical good value and \
    adapted to the batch size. [None]")
    classifier_options.add_argument('-P', type=str, dest="penalty", default=None, choices=['l1','l2','elasticnet'], help=\
                                        "Penalty term for SGD and other regularized linear models")
    classifier_options.add_argument('--l2r-alg', type=str, dest="l2r", default="listnet", choices=['listnet','adarank','ca', 'lambdamart'], help=\
                                        "L2R algorithm to use when classifier is 'listnet'")
    classifier_options.add_argument('--l2r-metric', type=str, dest="l2r_metric", default="ERR@k", choices=['MAP', 'NDCG', 'DCG', 'P', 'RR', 'ERR'], help=\
                                        "L2R metric to optimize for when using listnet classifier'")
    classifier_options.add_argument('--l2r-translation-prob', action="store_true", dest="translation_prob", default=False, help=
    "Whether to include the translation probability from concepts into titles. If set to true, number of jobs must be 1.")
    classifier_options.add_argument('--label_dependencies', action="store_true", dest="label_dependencies", default=False, help=
    "Whether the ClassifierStack should make use of all label information and thus take into account possible interdependencies.")
    classifier_options.add_argument('--l2r-neighbors', dest="l2r_neighbors", type=int, default=45, help=
    "Specify n_neighbors argument for KneighborsL2RClassifier.")
    classifier_options.add_argument('--batch_size', dest="batch_size", type=int, default=256, help=
    "Specify batch size for neural network training.")
    
    # neural network specific options
    neural_network_options = parser.add_argument_group("Neural Network Options")
    neural_network_options.add_argument('--dropout', type=float, dest="dropout", default=0.5, help=
    "Determine the keep probability for all dropout layers.")
    neural_network_options.add_argument('--memory', type=float, dest="memory", default=1.0, help=
    "Fraction of available GPU-memory to use for experiment.")
    neural_network_options.add_argument('--embedding_size', type=int, dest="embedding_size", default=300, help=
    "Determine the size of a word embedding vector (for MLP-Soph, CNN, and LSTM if embedding is learned jointly). \
    Specify --embedding_size=0 to skip the embedding layer, if applicable. [300]")
    neural_network_options.add_argument('--pretrained_embeddings', type=str, dest="pretrained_embeddings", default=None, help=
    "Specify the path to a file contraining pretrained word embeddings. The file must have a format where each line consists of the word\
     followed by the entries of its vectors, separated by blanks. If None is specified, the word embeddings are zero-initialized and trained\
     jointly with the classification task. [None]")
    neural_network_options.add_argument('--hidden_activation_function', type=str, dest="hidden_activation_function", default="relu", help=
    "Specify the activation function used on the hidden layers in MLP-Base and MLP-Soph. [relu]", choices = ["relu", "tanh", "identity", "swish"])
    neural_network_options.add_argument('--trainable_embeddings', action="store_true", dest="trainable_embeddings", default=False, help=
    "Whether to keep training the pretrained embeddings further with classification the task or not. [False]")
    neural_network_options.add_argument('--hidden_layers', type=int, dest="hidden_layers", nargs='+', default=[1000], help=
    "Specify the number of layers and the respective number of units as a list. The i-th element of the list \
    specifies the number of units in layer i. [1000]")
    neural_network_options.add_argument('--bottleneck_layers', type=int, dest="bottleneck_layers", nargs='+', default=None, help=
    "Specify the number of bottleneck layers and the respective number of units as a list. The i-th element of the list \
    specifies the number of units in layer i. In contrast to the --hidden_layers option, where the respective model decides\
    how to interprete multiple hidden layers, the bottleneck layers are feed forward layers which are pluged in between \
    the last layer of a particular model (e.g. CNN, LSTM) and the output layer. (None)")
    neural_network_options.add_argument('--standard_normal', action="store_true", dest="standard_normal", default=False, help=
    "Whether to normalize the input features to mean = 0 and std = 1 for MLPSoph. [False]")
    neural_network_options.add_argument('--batch_norm', action="store_true", dest="batch_norm", default=False, help=
    "Whether to apply batch normalization after at a hidden layer in MLP. [False]")
    neural_network_options.add_argument('--snn', action="store_true", dest="snn", default=False, help=
    "Whether to use SELU activation and -dropout. If set to False, the activation specified in --hidden_activation_function is used. [False]")
    neural_network_options.add_argument('--variational_recurrent_dropout', action="store_true", dest="variational_recurrent_dropout", default=False, help=
    "Whether to perform dropout on the recurrent unit between states in addition to dropout on the aggregated output. [False]")
    neural_network_options.add_argument('--bidirectional', action="store_true", dest="bidirectional", default=False, help=
    "When activated, we create two instances of (potentially multi-layered) LSTMs, where one reads the input from left to right and \
    the other reads it from right to left. [False]")
    neural_network_options.add_argument('--iterate_until_maxlength', action="store_true", dest="iterate_until_maxlength", default=False, help=
    "When activated, the LSTM always iterates max_features steps, even if the actual sequence is shorter. Instead, it consumes\
 at each additional step the padding symbol. The outputs of steps beyond the actual sequence length are taken into account as well for output aggregation. [False]")
    detailed_options.add_argument('--aggregate_output', type=str, dest='aggregate_output', default="average", help=
    "How to aggregate the outputs of an LSTM. 'last' uses the output at the last time step. 'average' takes the mean over all outputs. [average]", 
    choices = ["average", "last", "attention", "sum"])
    neural_network_options.add_argument('--optimize_threshold', action="store_true", dest="optimize_threshold", default=False, help=
    "Optimize the prediction threshold on validation set during training. [False]")
    neural_network_options.add_argument('--dynamic_max_pooling_p', type=int, dest="dynamic_max_pooling_p", default=1, help=
    "Specify the number of chunks (p) to perform max-pooling over. [1]")
    neural_network_options.add_argument('--meta_labeler_min_labels', type=int, dest="meta_labeler_min_labels", default=1, help=
    "Specify the minimum number of labels to assign the meta labeler can predict. [1]")
    neural_network_options.add_argument('--meta_labeler_max_labels', type=int, dest="meta_labeler_max_labels", default=None, help=
    "Specify the maximum number of labels to assign the meta labeler can predict. When 'None' is specified, the maximum \
    is computed from the data. [None]")
    detailed_options.add_argument('--meta_labeler_phi', type=str, dest='meta_labeler_phi', default=None, help=
    "Specify whether to predict number of labels from 'score' or from 'content', or whether to use meta labeler at all (None). [None]", 
    choices = ["content", "score"])
    neural_network_options.add_argument('--meta_labeler_alpha', type=float, dest="meta_labeler_alpha", default=0.1, help=
    "The alpha-weight of predicting the correct number of labels when doing meta-labeling. [0.1]")

    # persistence_options
    persistence_options = parser.add_argument_group("Feature Persistence Options")
    persistence_options.add_argument('-p', dest="persist", action="store_true", default=False, help=
    "Use persisted count vectors or persist if has changed.")
    persistence_options.add_argument('--repersist', dest="repersist", action="store_true", default=False, help=
    "Persisted features will be recalculated and overwritten.")
    persistence_options.add_argument('--persist_to', dest="persist_to", default=os.curdir + os.sep + 'persistence', help=
                                     "Path to persist files.")
    persistence_options.add_argument("--tf-model-path", dest="tf_model_path", default=".tmp_best_models", help=
                                     "Directory to store best models for early stopping.")

    return meta_parser, parser


def write_to_csv(score, opt):
    f = open(opt.output_file, 'a')
    if os.stat(opt.output_file).st_size == 0:
        for i, (key, _) in enumerate(opt.__dict__.items()):
            f.write(key + ";")
        for i, (key, _) in enumerate(score.items()):
            if i < len(score.items()) - 1:
                f.write(key + ";")
            else:
                f.write(key)
        f.write('\n')
        f.flush()
    f.close()

    f = open(opt.output_file, 'r')
    reader = csv.reader(f, delimiter=";")
    column_names = next(reader)
    f.close();

    f = open(opt.output_file, 'a')
    for i, key in enumerate(column_names):
        if i < len(column_names) - 1:
            if key in opt.__dict__:
                f.write(str(opt.__dict__[key]) + ";")
            else:
                f.write(str(score[key]) + ";")
        else:
            if key in opt.__dict__:
                f.write(str(opt.__dict__[key]))
            else:
                f.write(str(score[key]))
    f.write('\n')
    f.flush()
    f.close()

if __name__ == '__main__':
    meta_parser, parser = _generate_parsers()
    meta_args, remaining = meta_parser.parse_known_args()

    start_time = default_timer()

    if meta_args.config_file:
        lines = meta_args.config_file.readlines()
        n_executions = len(lines)
        for i, line in enumerate(lines):
            if line.startswith('#'): continue
            params = line.strip().split()
            args = parser.parse_args(remaining + params)
            if args.verbose: print("Line args:", args)
            if meta_args.dry: continue
            try:
                run(args)
            except Exception as e:
                print("Error while executing configuration of line %d:" % (i + 1), file=sys.stderr)
                exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
                traceback.print_exception(exceptionType, exceptionValue, exceptionTraceback, file=sys.stderr)

            progress = int(100 * (i + 1) / n_executions)
            sys.stdout.write("\r[%d%%] " % progress)
            sys.stdout.flush()

    else:
        args = parser.parse_args(remaining)
        if args.verbose: print("Args:", args)
        if not meta_args.dry: run(args)

    print('Duration: ' + str(datetime.timedelta(seconds=default_timer() - start_time)))
