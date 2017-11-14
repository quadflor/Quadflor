import numpy as np
import tensorflow as tf
from scipy.sparse.csr import csr_matrix
import scipy.sparse as sps
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelBinarizer
import math, numbers, os
from tensorflow.python.framework import ops, tensor_shape,  tensor_util
from tensorflow.python.ops import math_ops, random_ops, array_ops
from tensorflow.python.layers import utils
from datetime import datetime
from utils.tf_utils import tf_normalize, sequence_length, average_outputs, dynamic_max_pooling
#tf.logging.set_verbosity(tf.logging.INFO)

def _load_embeddings(filename, vocab_size, embedding_size):

    embeddings = np.zeros((vocab_size, embedding_size))
    with open(filename + ".tmp",'r') as embedding_file:

        i = 0
        for line in embedding_file.readlines():
            row = line.strip().split(' ')
            # omit escape sequences
            if len(row) != embedding_size + 1:
                continue
            else:
                embeddings[i, :] = np.asarray(row[1:], dtype=np.float32)
                i += 1
    return embeddings, embedding_size

def _embeddings(x_tensor, vocab_size, embedding_size, pretrained_embeddings = True, trainable_embeddings = True):
    
    if pretrained_embeddings:
        embedding_placeholder = tf.placeholder(tf.float32, shape=[vocab_size, embedding_size])
        lookup_table = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            trainable=trainable_embeddings, name="W")
        embedding_init = tf.assign(lookup_table, embedding_placeholder)
        
        embedded_words = tf.nn.embedding_lookup(lookup_table, x_tensor)
        return embedded_words, embedding_init, embedding_placeholder
         
    else:
        lookup_table = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                        name="W")
        embedded_words = tf.nn.embedding_lookup(lookup_table, x_tensor)
        return embedded_words
        

def _extract_vocab_size(X):
    # max index of vocabulary is encoded in last column
    vocab_size = X[0, -1] + 1

    # slice of the column from the input, as it's not part of the sequence
    sequence_length = X.shape[1]
    x_tensor = tf.placeholder(tf.int32, shape=(None, sequence_length), name = "x")
    feature_input = tf.slice(x_tensor, [0, 0], [-1, sequence_length - 1])
    return x_tensor, vocab_size, feature_input

def _init_embedding_layer(pretrained_embeddings_path, feature_input, embedding_size, vocab_size, 
                          params_fit, params_predict, trainable_embeddings, initializer_ops):
    if pretrained_embeddings_path is not None:
        embeddings, embedding_size = _load_embeddings(pretrained_embeddings_path, vocab_size, embedding_size)
        
        embedded_words, embedding_init, embedding_placeholder = _embeddings(feature_input, vocab_size, embedding_size, 
                                                            pretrained_embeddings = True,
                                                            trainable_embeddings = trainable_embeddings)
        initializer_ops.append((embedding_init, {embedding_placeholder : embeddings}))
        
    else:
        embedded_words = _embeddings(feature_input, vocab_size, embedding_size, 
                                     pretrained_embeddings = False, trainable_embeddings = trainable_embeddings)
        
    return embedded_words, embedding_size

def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res

def lstm_fn(X, y, keep_prob_dropout = 0.5, embedding_size = 30, hidden_layers = [1000], 
            aggregate_output = "average", 
            pretrained_embeddings_path = None,
            trainable_embeddings = True,
            variational_recurrent_dropout = True,
            bidirectional = False,
            iterate_until_maxlength = False):
    """Model function for LSTM."""
     
    x_tensor, vocab_size, feature_input = _extract_vocab_size(X)
    
    y_tensor = tf.placeholder(tf.float32, shape=(None, y.shape[1]), name = "y")
    dropout_tensor = tf.placeholder(tf.float32, name = "dropout")
    
    params_fit = {dropout_tensor : keep_prob_dropout}
    params_predict = {dropout_tensor : 1}

    if iterate_until_maxlength:
        # create a vector of correct shape and set to maxlen
        seq_length = tf.reduce_sum(feature_input, 1)
        seq_length = tf.cast(seq_length, tf.int32)
        seq_length = feature_input.get_shape().as_list()[1] * tf.ones_like(seq_length)
    else:
        seq_length = sequence_length(feature_input)
    
    initializer_operations = []
    
    embedded_words, _ = _init_embedding_layer(pretrained_embeddings_path, feature_input,
                                              embedding_size, vocab_size, 
                                              params_fit, 
                                              params_predict,
                                              trainable_embeddings,
                                              initializer_operations)
    
    def create_multilayer_lstm():
        # build multiple layers of lstms
        lstm_layers = []
        for hidden_layer_size in hidden_layers:
            single_lstm_layer = tf.contrib.rnn.LSTMCell(hidden_layer_size, use_peepholes = True)
            if variational_recurrent_dropout:
                single_lstm_layer = tf.contrib.rnn.DropoutWrapper(single_lstm_layer, 
                                                                  input_keep_prob=1., 
                                                                  output_keep_prob=1.,
                                                                  state_keep_prob=dropout_tensor,
                                                                  variational_recurrent=True,
                                                                  dtype = tf.float32)
            lstm_layers.append(single_lstm_layer)
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_layers)
        return stacked_lstm
    
    forward_lstm = create_multilayer_lstm()
    forward_state = forward_lstm.zero_state(tf.shape(embedded_words)[0], tf.float32)
    
    # bidirectional lstm?
    ## we can discard the state after the batch is fully processed
    if not bidirectional:
        output_state, _ = tf.nn.dynamic_rnn(forward_lstm, embedded_words, initial_state = forward_state, sequence_length = seq_length)
    else:
        backward_lstm = create_multilayer_lstm()
        backward_state = backward_lstm.zero_state(tf.shape(embedded_words)[0], tf.float32)
        bidi_output_states, _ = tf.nn.bidirectional_dynamic_rnn(forward_lstm, backward_lstm, embedded_words, 
                                                                initial_state_fw = forward_state, initial_state_bw = backward_state,
                                                                sequence_length = seq_length)
        ## we concatenate the outputs of forward and backward rnn in accordance with Hierarchical Attention Networks
        h1, h2 = bidi_output_states
        output_state = tf.concat([h1, h2], axis = 2, name = "concat_bidi_output_states")

    # note that dynamic_rnn returns zero outputs after seq_length
    if aggregate_output == "sum":
        output_state = tf.reduce_sum(output_state, axis = 1)
    elif aggregate_output == "average":
        output_state = average_outputs(output_state, seq_length)
    elif aggregate_output == "last":
        # return output at last time step
        output_state = extract_axis_1(output_state, seq_length - 1)
    elif aggregate_output == "attention":
        # we perform attention according to Hierarchical Attention Network (word attention)
        ## compute hidden representation of outputs
        ### context vector size as in HN-ATT
        context_vector_size = 100
        hidden_output_representation = tf.contrib.layers.fully_connected(output_state, context_vector_size, activation_fn = tf.nn.tanh)
        ## compute dot product with context vector
        context_vector = tf.Variable(tf.random_normal([context_vector_size], stddev=0.1))
        dot_product = tf.tensordot(hidden_output_representation, context_vector, [[2], [0]])
        ## compute weighted sum
        attention_weights = tf.nn.softmax(dot_product)
        attention_weights = tf.expand_dims(attention_weights, -1)
        output_state = tf.reduce_sum(tf.multiply(output_state, attention_weights), axis = 1)
    else:
        raise ValueError("Aggregation method not implemented!")
        
    hidden_layer = tf.nn.dropout(output_state, dropout_tensor)
    
    return x_tensor, y_tensor, hidden_layer, params_fit, params_predict, initializer_operations
def cnn_fn(X, y, keep_prob_dropout = 0.5, embedding_size = 30, hidden_layers = [1000], 
           pretrained_embeddings_path = None,
           trainable_embeddings = True,
           dynamic_max_pooling_p = 1,
           # these are set according to Kim's Sentence Classification
           window_sizes = [3, 4, 5],
           num_filters = 100):
    """Model function for CNN."""
    
    # x_tensor includes the max_index_column, feature_input doesnt. go on with feature_input, but return x_tensor for feed_dict
    x_tensor, vocab_size, feature_input = _extract_vocab_size(X)
    max_length = X.shape[1] - 1

    y_tensor = tf.placeholder(tf.float32, shape=(None, y.shape[1]), name = "y")
    dropout_tensor = tf.placeholder(tf.float32, name = "dropout")
    
    params_fit = {dropout_tensor : keep_prob_dropout}
    params_predict = {dropout_tensor : 1}

    seq_length = sequence_length(feature_input)
    seq_length = tf.cast(seq_length, tf.float32)
    
    initializer_operations = []
    embedded_words, embedding_size = _init_embedding_layer(pretrained_embeddings_path, 
                                                           feature_input, embedding_size, 
                                                           vocab_size, params_fit, 
                                                           params_predict,
                                                           trainable_embeddings,
                                                           initializer_operations)
    
    # need to extend the number of dimensions here in order to use the predefined pooling operations, which assume 2d pooling
    embedded_words = tf.expand_dims(embedded_words, -1)
    
    stride = [1, 1, 1, 1]
    padding = "VALID"
    
    pooled_outputs = []
    for window_size in window_sizes:
        filter_weights = tf.Variable(tf.random_normal([window_size, embedding_size, 1, num_filters], stddev=0.1))
        conv = tf.nn.conv2d(embedded_words, filter_weights, stride, padding)
        bias = tf.Variable(tf.random_normal([num_filters]))
        detector = tf.nn.relu(tf.nn.bias_add(conv, bias))
        
        concatenated_pooled_chunks = dynamic_max_pooling(detector, seq_length, max_length, num_filters, window_size, dynamic_max_pooling_p = dynamic_max_pooling_p)
        pooled_outputs.append(concatenated_pooled_chunks)
        
    concatenated_pools = tf.concat(pooled_outputs, 1)
    num_filters_total = num_filters * len(window_sizes) * dynamic_max_pooling_p
    hidden_layer = tf.reshape(concatenated_pools, [-1, num_filters_total])
    
    hidden_layer = tf.nn.dropout(hidden_layer, dropout_tensor)
    
    return x_tensor, y_tensor, hidden_layer, params_fit, params_predict, initializer_operations

def mlp_base_fn(X, y, keep_prob_dropout = 0.5, hidden_activation_function = tf.nn.relu):
    """Model function for MLP-Soph."""
    # convert sparse tensors to dense
    x_tensor = tf.placeholder(tf.float32, shape=(None, X.shape[1]), name = "x")
    y_tensor = tf.placeholder(tf.float32, shape=(None, y.shape[1]), name = "y")
    dropout_tensor = tf.placeholder(tf.float32, name = "dropout")
    
    params_fit = {dropout_tensor : keep_prob_dropout}
    params_predict = {dropout_tensor : 1}
    
    # Connect the first hidden layer to input layer
    # (features) with relu activation and add dropout
    hidden_layer = tf.contrib.layers.fully_connected(x_tensor, 1000, activation_fn = hidden_activation_function)
    hidden_dropout = tf.nn.dropout(hidden_layer, dropout_tensor)
    
    return x_tensor, y_tensor, hidden_dropout, params_fit, params_predict, []

# https://github.com/bioinf-jku/SNNs/blob/master/selu.py
def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=1337, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        alpha.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = math_ops.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * math_ops.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))

# https://github.com/bioinf-jku/SNNs/blob/master/selu.py
def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def mlp_soph_fn(X, y, keep_prob_dropout = 0.5, embedding_size = 30, hidden_layers = [1000], self_normalizing = False, hidden_activation_function = tf.nn.relu,
                standard_normal = False, batch_norm = True):
    """Model function for MLP-Soph."""
    
    # convert sparse tensors to dense
    x_tensor = tf.placeholder(tf.float32, shape=(None, X.shape[1]), name = "x")
    y_tensor = tf.placeholder(tf.float32, shape=(None, y.shape[1]), name = "y")
    dropout_tensor = tf.placeholder(tf.float32, name = "dropout")
    
    params_fit = {dropout_tensor : keep_prob_dropout}
    params_predict = {dropout_tensor : 1}
    
    # we need to have the input data scaled such they have mean 0 and variance 1
    if standard_normal:
        scaled_input = tf_normalize(X, x_tensor)
    else:
        scaled_input = x_tensor
    
    # apply a look-up as described by the fastText paper
    if embedding_size > 0:
        lookup_table = tf.Variable(tf.truncated_normal([X.shape[1], embedding_size], mean=0.0, stddev=0.1))
        embedding_layer = tf.matmul(scaled_input, lookup_table)
    else:
        embedding_layer = scaled_input
    
    # Connect the embedding layer to the hidden layers
    # (features) with relu activation and add dropout everywhere
    hidden_layer = embedding_layer
    for hidden_units in hidden_layers:
        if not self_normalizing:
            normalizer_fn = tf.layers.batch_normalization if batch_norm else None  
            hidden_layer = tf.contrib.layers.fully_connected(hidden_layer, hidden_units, activation_fn = hidden_activation_function, normalizer_fn = normalizer_fn)
            hidden_layer = tf.nn.dropout(hidden_layer, dropout_tensor)
        else:
            hidden_layer = tf.contrib.layers.fully_connected(hidden_layer, hidden_units,
                                                      activation_fn=None,
                                                      weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
            hidden_layer = selu(hidden_layer)
            # dropout_selu expects to be given the dropout rate instead of keep probability
            hidden_layer = dropout_selu(hidden_layer, tf.constant(1, tf.float32) - dropout_tensor)
    
    return x_tensor, y_tensor, hidden_layer, params_fit, params_predict, []

def swish(x):
    """
    Implementation of Swish: a Self-Gated Activation Function (https://arxiv.org/abs/1710.05941)
    """
    return x * tf.sigmoid(x)

def _transform_activation_function(func):
    if func == "relu":
        hidden_activation_function = tf.nn.relu
    elif func == "tanh":
        hidden_activation_function = tf.nn.tanh
    elif func == "identity":
        hidden_activation_function = tf.identity
    elif func == "swish":
        hidden_activation_function = swish
    return hidden_activation_function

def mlp_base(hidden_activation_function = "relu"):
    hidden_activation_function = _transform_activation_function(hidden_activation_function)
    
    return lambda X, y : mlp_soph_fn(X, y, keep_prob_dropout = 0.5, embedding_size = 0, 
                                     hidden_layers = [1000], self_normalizing = False,
                                     hidden_activation_function=hidden_activation_function,
                                     standard_normal = False, batch_norm = False)
    
def mlp_soph(keep_prob_dropout, embedding_size, hidden_layers, self_normalizing, standard_normal, batch_norm, hidden_activation_function = "relu"):
    hidden_activation_function = _transform_activation_function(hidden_activation_function)
    
    return lambda X, y : mlp_soph_fn(X, y, keep_prob_dropout = keep_prob_dropout, embedding_size = embedding_size, 
                                     hidden_layers = hidden_layers, self_normalizing = self_normalizing,
                                     hidden_activation_function=hidden_activation_function,
                                     standard_normal = standard_normal, batch_norm = batch_norm)
    
def cnn(keep_prob_dropout, embedding_size, hidden_layers, pretrained_embeddings_path, trainable_embeddings, dynamic_max_pooling_p, window_sizes, num_filters):
    
    return lambda X, y : cnn_fn(X, y, keep_prob_dropout = keep_prob_dropout, embedding_size = embedding_size, 
                                     hidden_layers = hidden_layers, pretrained_embeddings_path=pretrained_embeddings_path,
                                     trainable_embeddings = trainable_embeddings,
                                     dynamic_max_pooling_p = dynamic_max_pooling_p,
                                     window_sizes = window_sizes,
                                     num_filters = num_filters)
    
def lstm(keep_prob_dropout, embedding_size, hidden_layers, pretrained_embeddings_path, trainable_embeddings, variational_recurrent_dropout,
         bidirectional, aggregate_output, iterate_until_maxlength):
        
    return lambda X, y : lstm_fn(X, y, keep_prob_dropout = keep_prob_dropout, embedding_size = embedding_size, 
                                     hidden_layers = hidden_layers, pretrained_embeddings_path=pretrained_embeddings_path,
                                     trainable_embeddings = trainable_embeddings,
                                     variational_recurrent_dropout=variational_recurrent_dropout,
                                     bidirectional = bidirectional,
                                     aggregate_output = aggregate_output,
                                     iterate_until_maxlength = iterate_until_maxlength)


class BatchGenerator:
    
    def __init__(self, X, y, batch_size, shuffle, predict):
        self.X = X
        self.y = y
        self.number_of_batches = np.ceil(X.shape[0] / batch_size)
        self.counter = 0
        self.sample_index = np.arange(X.shape[0])
        self.batch_size = batch_size
        self.predict = predict
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.sample_index)
            

    def _batch_generator(self):
        
        batch_index = self.sample_index[self.batch_size * self.counter:self.batch_size * (self.counter + 1)]
        
        X_batch = self.X[batch_index, :]
        if sps.issparse(X_batch):
            X_batch = X_batch.toarray()
            
        if not self.predict:
            y_batch = self.y[batch_index].toarray()
        self.counter += 1
        if self.counter == self.number_of_batches:
            if self.shuffle:
                np.random.shuffle(self.sample_index)
            self.counter = 0
        if not self.predict:
            return X_batch, y_batch
        else:
            return X_batch
    

class MultiLabelSKFlow(BaseEstimator):
    """
    This is a wrapper class for tf.contrib.learn.Estimators, so it adheres to the fit/predict naming conventions of sk-learn.
    It already handles mini-batching, whose behavior can be controlled by providing the respective parameters to the init function.
    
    The concrete TensorFlow model to execute can be specified in terms of the 'get_model' function.
    This function in turn has to return a 'model_fn' function and a 'params' dictionary.
    These arguments will passed to the tf.contrib.learn. Estimator class and have thus to conform the formats
    described in 'https://www.tensorflow.org/extend/estimators'.
    On top of that, 'model_fn' has to accept an additional, non-positional argument 'num_classes' which is used to 
    infer the output size. Furthermore, the function has to assume the 'features' and 'targets' parameters to be of the Tensor class.
    """
    
    def __init__(self, batch_size = 5, num_epochs = 10, get_model = mlp_base(), threshold = 0.2, learning_rate = 0.1, patience = 5,
                       validation_metric = lambda y1, y2 : f1_score(y1, y2, average = "samples"), 
                       optimize_threshold = True,
                       threshold_window = np.linspace(-0.03, 0.03, num=7),
                       tf_model_path = ".tmp_best_models",
                       num_steps_before_validation = None,
                       hidden_activation_function = tf.nn.relu,
                       bottleneck_layers = None,
                       hidden_keep_prob = 0.5,
                       gpu_memory_fraction = 1.,
                       meta_labeler_phi = None,
                       meta_labeler_alpha = 0.1,
                       meta_labeler_min_labels = 1,
                       meta_labeler_max_labels = None):
        """
    
        """
        
        self.get_model = get_model
        
        # enable early stopping on validation set
        self.validation_data_position = None
        self.num_steps_before_validation = num_steps_before_validation
        
        # configurations for bottleneck layers
        self.hidden_activation_function = hidden_activation_function
        self.bottleneck_layers = bottleneck_layers
        self.hidden_keep_prob = hidden_keep_prob
        
        # configuration for meta-labeler
        self.meta_labeler_phi = meta_labeler_phi
        self.meta_labeler_alpha = meta_labeler_alpha
        self.num_label_binarizer = None
        self.meta_labeler_max_labels = meta_labeler_max_labels
        self.meta_labeler_min_labels = meta_labeler_min_labels
        
        # used by this class
        self.validation_metric = validation_metric
        self.optimize_threshold = optimize_threshold
        self.threshold_window = threshold_window
        self.patience = patience
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.threshold = threshold
        if learning_rate is None:
            self.learning_rate = self.batch_size / 512 * 0.01
        else:
            self.learning_rate = learning_rate
        
        # path to save the tensorflow model to
        self.TF_MODEL_PATH = tf_model_path
        self._save_model_path = self._get_save_model_path()
        
        # determine how much of gpu to use
        self.gpu_memory_fraction = gpu_memory_fraction
        
    def _get_save_model_path(self):
        TMP_FOLDER = self.TF_MODEL_PATH
        if not os.path.exists(TMP_FOLDER):
            os.makedirs(TMP_FOLDER)
        return TMP_FOLDER + "/best-model-" + self.get_model.__name__ + str(datetime.now())
       
    def _calc_num_steps(self, X):
        return int(np.ceil(X.shape[0] / self.batch_size))


    def _predict_batch(self, X_batch):
        feed_dict = {self.x_tensor: X_batch}
        feed_dict.update(self.params_predict)
        
        if self.meta_labeler_phi is None: 
            predictions = self.session.run(self.predictions, feed_dict = feed_dict)
        else:
            predictions = self.session.run([self.predictions, self.meta_labeler_prediction], feed_dict = feed_dict)
            
        return predictions
    
    def _make_binary_decision(self, predictions):
        if self.meta_labeler_phi is None:
            y_pred = predictions > self.threshold
        else:
            #TODO: implement meta-labeler label assignment
            predictions, meta_labeler_predictions = predictions
            max_probability_cols = np.argmax(meta_labeler_predictions, axis = 1)
            max_probability_indices = tuple(np.indices([meta_labeler_predictions.shape[0]]))+(max_probability_cols,)
            meta_labeler_predictions = np.zeros_like(meta_labeler_predictions)
            meta_labeler_predictions[max_probability_indices] = 1
            meta_labeler_predictions = self.num_label_binarizer.inverse_transform(meta_labeler_predictions, 0)
            y_pred = np.zeros_like(predictions)
            for i in range(predictions.shape[0]):
                num_labels_for_sample = meta_labeler_predictions[i]
                top_indices = (-predictions[i,:]).argsort()[:num_labels_for_sample]
                y_pred[i,top_indices] = 1
                
        return csr_matrix(y_pred)
        
    def _compute_validation_score(self, session, X_val_batch, y_val_batch):

        feed_dict = {self.x_tensor: X_val_batch}
        feed_dict.update(self.params_predict)

        if self.validation_metric == "val_loss":
            return session.run(self.loss, feed_dict = feed_dict)

        elif callable(self.validation_metric):
            predictions = self._predict_batch(X_val_batch)
            y_pred = self._make_binary_decision(predictions)
            if self.optimize_threshold:
                return self.validation_metric(y_val_batch, y_pred), predictions
            else:
                return self.validation_metric(y_val_batch, y_pred)

    def _print_progress(self, epoch, batch_i, steps_per_epoch, avg_validation_score, best_validation_score, total_loss, meta_loss, label_loss):
        
        progress_string = 'Epoch {:>2}/{:>2}, Batch {:>2}/{:>2}, Loss: {:0.4f}, Validation-Score: {:0.4f}, Best Validation-Score: {:0.4f}'
        format_parameters = [epoch + 1, self.num_epochs, batch_i + 1, steps_per_epoch, 
                                                     total_loss, avg_validation_score, best_validation_score]
        if self.meta_labeler_phi is None:
            progress_string += ', Threshold: {:0.2f}'
            format_parameters.append(self.threshold)
            
        else: 
            progress_string += ', Label-Loss: {:0.4f}, Meta-Loss: {:0.4f}'
            format_parameters.extend([label_loss, meta_loss])
        
        progress_string = progress_string.format(*format_parameters)
        print(progress_string, end='\r')
            
    def _num_labels_discrete(self, y, min_number_labels = 1, max_number_labels = None):
        """
        Counts for each row in 'y' how many of the columns are set to 1. Outputs the result in turn as a binary indicator matrix where
        the columns 0, ..., m correspond to 'min_number_labels', 'min_number_labels' + 1, ..., 'max_number_labels'.  
        
        Parameters
        ----------
        y: (sparse) numpy array of shape [n_samples, n_classes]
            An indicator matrix denoting which classes are assigned to a sample (multiple columns per row may be 1)
        min_number_labels: int, default=1
            Minimum number of labels each sample has to have. If a sample has less than 'min_number_labels' assigned,
            the corresponding output is set to 'min_number_labels'.
        max_number_labels: int, default=None
            Maximum number of labels each sample has to have. If a sample has more than 'min_number_labels' assigned,
            the corresponding output is set to 'max_number_labels'. If 'max_number_labels' is None, it is set to the max number found
            in y.
        Returns
        ---------
        num_samples_y: (sparse) numpy array of shape [n_samples, max_number_samples - min_number_samples + 1]
        """
        
        num_samples_y = np.array(np.sum(y, axis = 1))
        num_samples_y = num_samples_y.reshape(-1)
        num_samples_y[num_samples_y < min_number_labels] = min_number_labels
        
        if max_number_labels is None:
            max_number_labels = np.max(num_samples_y)
            
        num_samples_y[num_samples_y > max_number_labels] = max_number_labels
        
        # 'fit' method calls this
        if self.num_label_binarizer is None:
            self.num_label_binarizer = LabelBinarizer()
            self.num_label_binarizer.fit(num_samples_y)
        
        indicator_matrix_num_labels = self.num_label_binarizer.transform(num_samples_y)
        return indicator_matrix_num_labels
        

    def fit(self, X, y):
        self.y = y
        
        val_pos = self.validation_data_position
        
        if val_pos is not None:
            X_train, y_train, X_val, y_val = X[:val_pos, :], y[:val_pos,:], X[val_pos:, :], y[val_pos:, :]
         
            validation_batch_generator = BatchGenerator(X_val, y_val, self.batch_size, False, False)
            validation_predictions = self._calc_num_steps(X_val)
            steps_per_epoch = self._calc_num_steps(X_train)
            
            # determine after how many batches to perform validation
            num_steps_before_validation = self.num_steps_before_validation
            if self.num_steps_before_validation is None:
                num_steps_before_validation = steps_per_epoch
            num_steps_before_validation = int(min(steps_per_epoch, num_steps_before_validation))
        else:
            steps_per_epoch = self._calc_num_steps(X)
            X_train = X
            y_train = y
         
        # Remove previous weights, bias, inputs, etc..
        tf.reset_default_graph()
        tf.set_random_seed(1337)
                
        # Inputs
        
        # get_model has to return a 
        self.x_tensor, self.y_tensor, self.last_layer, self.params_fit, self.params_predict, initializer_operations = self.get_model(X, y)
        
        # add bottleneck layer
        if self.bottleneck_layers is not None:
            bottleneck_dropout_tensor = tf.placeholder(tf.float32, name = "bottleneck_dropout")
            self.params_fit.update({bottleneck_dropout_tensor : self.hidden_keep_prob})
            self.params_predict.update({bottleneck_dropout_tensor : 1})
            for units in self.bottleneck_layers:
                self.last_layer = tf.contrib.layers.fully_connected(self.last_layer, units, activation_fn = self.hidden_activation_function)
                self.last_layer = tf.nn.dropout(self.last_layer, bottleneck_dropout_tensor)
                
        
        # Name logits Tensor, so that is can be loaded from disk after training
        #logits = tf.identity(logits, name='logits')
        logits = tf.contrib.layers.linear(self.last_layer,
                                                num_outputs=y.shape[1])
                
        
        # Loss and Optimizer
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y_tensor)
        loss = tf.reduce_sum(losses, axis = 1)
        self.label_loss = tf.reduce_mean(loss, axis = 0)
        
        # prediction
        self.predictions = tf.sigmoid(logits)
        
        if self.meta_labeler_phi is not None:
            
            # compute target of meta labeler
            y_num_labels = self._num_labels_discrete(y_train, min_number_labels=self.meta_labeler_min_labels,max_number_labels=self.meta_labeler_max_labels)
            y_num_labels_tensor = tf.placeholder(tf.float32, shape=(None, y_num_labels.shape[1]), name = "y_num_labels")
            
            # compute logits of meta labeler
            if self.meta_labeler_phi == "content":
                meta_logits = tf.contrib.layers.linear(self.last_layer, num_outputs=y_num_labels.shape[1])
            elif self.meta_labeler_phi == "score":
                meta_logits = tf.contrib.layers.linear(self.predictions, num_outputs=y_num_labels.shape[1])
            
            # compute loss of meta labeler
            meta_labeler_loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_num_labels_tensor, logits = meta_logits)
            self.meta_labeler_loss = tf.reduce_mean(meta_labeler_loss, axis = 0)
            
            # compute prediction of meta labeler
            self.meta_labeler_prediction = tf.nn.softmax(meta_logits)
            
            # add meta labeler loss to labeling loss
            self.loss = (1 - self.meta_labeler_alpha) * self.label_loss + self.meta_labeler_alpha * self.meta_labeler_loss
        else:
            self.loss = self.label_loss
        
        # optimize
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.session = session
        # Initializing the variables
        session.run(tf.global_variables_initializer())
        for (init_op, init_op_feed_dict) in initializer_operations:
            session.run(init_op, feed_dict = init_op_feed_dict)
        
        batch_generator = BatchGenerator(X_train, y_train, self.batch_size, True, False)
        # Training cycle
        objective = 1 if self.validation_metric == "val_loss" else -1
        avg_validation_score = math.inf * objective
        best_validation_score = math.inf * objective
        epochs_of_no_improvement = 0
        most_consecutive_epochs_with_no_improvement = 0
        batches_counter = 0
        epoch = 0
        stop_early = False
        while epoch < self.num_epochs and not stop_early:
            
            if val_pos is not None and epochs_of_no_improvement == self.patience:
                break
            
            # Loop over all batches
            for batch_i in range(steps_per_epoch):
                X_batch, y_batch = batch_generator._batch_generator()
                feed_dict = {self.x_tensor: X_batch, self.y_tensor: y_batch}
                feed_dict.update(self.params_fit)
                
                if self.meta_labeler_phi is not None:
                    feed_dict.update({y_num_labels_tensor : self._num_labels_discrete(y_batch)})
                
                session.run(optimizer, feed_dict = feed_dict)

                # overwrite parameter values for prediction step
                feed_dict.update(self.params_predict)
                
                # compute losses to track progress
                if self.meta_labeler_phi is not None:
                    total_loss, label_loss, meta_loss = session.run([self.loss, self.label_loss, self.meta_labeler_loss], feed_dict = feed_dict)
                else:
                    total_loss = session.run(self.loss, feed_dict = feed_dict)
                    label_loss, meta_loss = None, None
                
                batches_counter += 1
                is_last_epoch = epoch == self.num_epochs - 1
                is_last_batch_in_epoch = batch_i == steps_per_epoch - 1
                # calculate validation loss at end of epoch if early stopping is on
                if val_pos is not None and (batches_counter == num_steps_before_validation
                    or (is_last_epoch and is_last_batch_in_epoch)):
                    
                    batches_counter = 0
                    
                    validation_scores = []
                    weights = []
                    
                    # save predictions so we can optimize threshold later
                    val_predictions = np.zeros((X_val.shape[0], self.y.shape[1]))
                    for i in range(validation_predictions):
                        X_val_batch, y_val_batch = validation_batch_generator._batch_generator()
                        weights.append(X_val_batch.shape[0])

                        if self.optimize_threshold:
                            batch_val_score, val_predictions[i * self.batch_size:(i+1) * self.batch_size, :] = self._compute_validation_score(session, X_val_batch, y_val_batch)
                        else:
                            batch_val_score = self._compute_validation_score(session, X_val_batch, y_val_batch)
                        validation_scores.append(batch_val_score)
                    avg_validation_score = np.average(np.array(validation_scores), weights = np.array(weights))

                    if self.optimize_threshold:
                        best_score = -1 * math.inf
                        best_threshold = self.threshold
                        for t_diff in self.threshold_window:
                            t = self.threshold + t_diff
                            score = self.validation_metric(y_val, csr_matrix(val_predictions > t))
                            if score > best_score:
                                best_threshold = t
                                best_score = score

                    is_better_score = avg_validation_score < best_validation_score if objective == 1 else avg_validation_score > best_validation_score
                    if is_better_score:
                        # save model
                        # Save model for prediction step
                        best_validation_score = avg_validation_score
                        saver = tf.train.Saver()
                        saver.save(session, self._save_model_path)
                        
                        if most_consecutive_epochs_with_no_improvement < epochs_of_no_improvement:
                            most_consecutive_epochs_with_no_improvement = epochs_of_no_improvement
                        epochs_of_no_improvement = 0

                        # save the threshold at best model, too.
                        if self.optimize_threshold:
                            self.threshold = best_threshold
                    else:
                        epochs_of_no_improvement += 1
                        if epochs_of_no_improvement > self.patience:
                            print("No improvement in validation loss for", self.patience, "epochs. Stopping early.")
                            stop_early = True
                            break

                # print progress
                self._print_progress(epoch, batch_i, steps_per_epoch, avg_validation_score, best_validation_score, total_loss, meta_loss, label_loss)
                
            epoch += 1
            
        print('')
        
        print("Training of TensorFlow model finished!")
        print("Longest sequence of epochs of no improvement:", most_consecutive_epochs_with_no_improvement)
    
        #def model_fn_with_num_classes(features, targets, mode, params):
        #   return self.model_fn(features, targets, mode, params, y.shape[1])
        
        #self._estimator = tf.contrib.learn.Estimator(model_fn=model_fn_with_num_classes, params=self.model_params)
        
        #=======================================================================
        # val_pos = self.validation_data_position
        # monitors = []
        # if val_pos is not None:
        #     #callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto'))
        #     #callbacks.append(ModelCheckpoint("weights.best.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min'))
        #     X_train, y_train, X_val, y_val = X[:val_pos, :], y[:val_pos,:], X[val_pos:, :], y[val_pos:,:]
        # 
        #     validation_batch_generator = BatchGenerator(X_val, y_val, X_val.shape[0], False, False)
        #     steps_per_epoch = int(np.ceil(X_train.shape[0] / self.batch_size))
        #     monitors.append(tf.contrib.learn.monitors.ValidationMonitor(input_fn = validation_batch_generator._batch_generator,
        #                                                                  every_n_steps=steps_per_epoch,
        #                                                                  early_stopping_metric="loss",
        #                                                                  early_stopping_rounds=5 * steps_per_epoch,
        #                                                                  eval_steps = int(np.ceil(X_val.shape[0] / self.batch_size))))
        # else:
        #     steps_per_epoch = int(np.ceil(X.shape[0] / self.batch_size))
        #     X_train = X
        #     y_train = y
        #     
        # batch_generator = BatchGenerator(X_train, y_train, self.batch_size, True, False)
        # 
        # self._estimator.fit(input_fn= batch_generator._batch_generator, steps=self.num_epochs * steps_per_epoch, monitors = monitors)
        #=======================================================================
        
        
    def predict(self, X):
        
        session = self.session
        #loaded_graph = tf.Graph()
        if self.validation_data_position:
            # Load model
            loader = tf.train.import_meta_graph(self._save_model_path + '.meta')
            loader.restore(self.session, self._save_model_path)

        # Get Tensors from loaded model
        #loaded_x = loaded_graph.get_tensor_by_name('x:0')
        #loaded_y = loaded_graph.get_tensor_by_name('y:0')
        #loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        #loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        #loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0'
        
        prediction = np.zeros((X.shape[0], self.y.shape[1]))
        batch_generator = BatchGenerator(X, None, self.batch_size, False, True)
        prediction_steps = self._calc_num_steps(X)
        for i in range(prediction_steps):
            X_batch = batch_generator._batch_generator()
            preds = self._predict_batch(X_batch)
            binary_decided_preds = self._make_binary_decision(preds)
            prediction[i * self.batch_size:(i+1) * self.batch_size, :] = binary_decided_preds.todense()
        
        result = csr_matrix(prediction)
        
        # close the session, since no longer needed
        session.close()
        return result


