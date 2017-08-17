import numpy as np
import tensorflow as tf
from scipy.sparse.csr import csr_matrix
import scipy.sparse as sps
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
import math, numbers, os
from tensorflow.python.framework import ops, tensor_shape,  tensor_util
from tensorflow.python.ops import math_ops, random_ops, array_ops
from tensorflow.python.layers import utils
from datetime import datetime
#tf.logging.set_verbosity(tf.logging.INFO)

def _load_embeddings(filename, vocab_size, embedding_size):

    embeddings = np.zeros((vocab_size, embedding_size))
    with open(filename,'r') as embedding_file:
        # skip first line, which we dont need
        embedding_file.readline()

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
    
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        
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

def lstm_fn(X, y, keep_prob_dropout = 0.5, embedding_size = 30, hidden_layers = [1000], 
            aggregate_output = True, 
            pretrained_embeddings_path = None,
            trainable_embeddings = True):
    """Model function for LSTM."""
     
    x_tensor, vocab_size, feature_input = _extract_vocab_size(X)
    
    y_tensor = tf.placeholder(tf.float32, shape=(None, y.shape[1]), name = "y")
    dropout_tensor = tf.placeholder(tf.float32, name = "dropout")
    
    params_fit = {dropout_tensor : keep_prob_dropout}
    params_predict = {dropout_tensor : 1}
    
    initializer_operations = []
    
    embedded_words, _ = _init_embedding_layer(pretrained_embeddings_path, feature_input,
                                              embedding_size, vocab_size, 
                                              params_fit, 
                                              params_predict,
                                              trainable_embeddings,
                                              initializer_operations)
    
    # build multiple layers of lstms
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(hidden_layer_size) for hidden_layer_size in hidden_layers])
    state = stacked_lstm.zero_state(tf.shape(embedded_words)[0], tf.float32)
    #state = tf.zeros([tf.shape(embedded_words)[0].value, stacked_lstm.state_size])
    
    # we can discard the state after the batch is fully processed
    output_state, _ = tf.nn.dynamic_rnn(stacked_lstm, embedded_words, initial_state = state)

    if aggregate_output:
        output_state = tf.reduce_mean(output_state, axis = 1)
    
    hidden_layer = tf.nn.dropout(output_state, dropout_tensor)
    
    return x_tensor, y_tensor, hidden_layer, params_fit, params_predict, initializer_operations
def cnn_fn(X, y, keep_prob_dropout = 0.5, embedding_size = 30, hidden_layers = [1000], 
           pretrained_embeddings_path = None,
           trainable_embeddings = True):
    """Model function for CNN."""
    
    # x_tensor includes the max_index_column, feature_input doesnt. go on with feature_input, but return x_tensor for feed_dict
    x_tensor, vocab_size, feature_input = _extract_vocab_size(X)
    sequence_length = X.shape[1] - 1

    y_tensor = tf.placeholder(tf.float32, shape=(None, y.shape[1]), name = "y")
    dropout_tensor = tf.placeholder(tf.float32, name = "dropout")
    
    params_fit = {dropout_tensor : keep_prob_dropout}
    params_predict = {dropout_tensor : 1}
    
    initializer_operations = []
    embedded_words, embedding_size = _init_embedding_layer(pretrained_embeddings_path, 
                                                           feature_input, embedding_size, 
                                                           vocab_size, params_fit, 
                                                           params_predict,
                                                           trainable_embeddings,
                                                           initializer_operations)
    
    # need to extend the number of dimensions here in order to use the predefined pooling operations, which assume 2d pooling
    embedded_words = tf.expand_dims(embedded_words, -1)
    # these are set according to Kim's Sentence Classification
    window_sizes = [3, 4, 5]
    num_filters = 100
    stride = [1, 1, 1, 1]
    padding = "VALID"
    
    pooled_outputs = []
    for window_size in window_sizes:
        filter_weights = tf.Variable(tf.random_normal([window_size, embedding_size, 1, num_filters], stddev=0.1))
        conv = tf.nn.conv2d(embedded_words, filter_weights, stride, padding)
        bias = tf.Variable(tf.random_normal([num_filters]))
        detector = tf.nn.relu(tf.nn.bias_add(conv, bias))
        pooling = tf.nn.max_pool(detector,
                                ksize = [1, sequence_length - window_size + 1, 1, 1],
                                strides = stride,
                                padding = "VALID")
        pooled_outputs.append(tf.reshape(pooling, [-1, num_filters]))
        
 
    concatenated_pools = tf.concat(pooled_outputs, 1)
    num_filters_total = num_filters * len(window_sizes)
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
                standard_normal = False):
    """Model function for MLP-Soph."""
    
    # convert sparse tensors to dense
    x_tensor = tf.placeholder(tf.float32, shape=(None, X.shape[1]), name = "x")
    y_tensor = tf.placeholder(tf.float32, shape=(None, y.shape[1]), name = "y")
    dropout_tensor = tf.placeholder(tf.float32, name = "dropout")
    
    params_fit = {dropout_tensor : keep_prob_dropout}
    params_predict = {dropout_tensor : 1}
    
    # we need to have the input data scaled such they have mean 0 and variance 1
    if self_normalizing or standard_normal:
        m = X.mean(axis = 0)
        
        # because we are dealing with sparse matrices, we need to compute the variance as
        # E[X^2] - E[X]^2
        X_square = X.power(2)
        m_square = X_square.mean(axis = 0)
        v = m_square - np.power(m, 2)
        s = np.sqrt(v)
        
        
        m = tf.constant(m, dtype = tf.float32) 
        s = tf.constant(s, dtype = tf.float32)
        
        scaled_input = (x_tensor - m) / s
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
            hidden_layer = tf.contrib.layers.fully_connected(hidden_layer, hidden_units, activation_fn = hidden_activation_function)
            hidden_layer = tf.nn.dropout(hidden_layer, dropout_tensor)
        else:
            hidden_layer = tf.contrib.layers.fully_connected(hidden_layer, hidden_units,
                                                      activation_fn=None,
                                                      weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
            hidden_layer = selu(hidden_layer)
            # dropout_selu expects to be given the dropout rate instead of keep probability
            hidden_layer = dropout_selu(hidden_layer, tf.constant(1, tf.float32) - dropout_tensor)
    
    return x_tensor, y_tensor, hidden_layer, params_fit, params_predict, []

def _transform_activation_function(func):
    if func == "relu":
        hidden_activation_function = tf.nn.relu
    elif func == "tanh":
        hidden_activation_function = tf.nn.tanh
    return hidden_activation_function

def mlp_base(keep_prob_dropout, hidden_activation_function = "relu"):
    hidden_activation_function = _transform_activation_function(hidden_activation_function)
    
    return lambda X, y : mlp_base_fn(X, y, keep_prob_dropout = keep_prob_dropout, hidden_activation_function=hidden_activation_function)

def mlp_soph(keep_prob_dropout, embedding_size, hidden_layers, self_normalizing, standard_normal, hidden_activation_function = "relu"):
    hidden_activation_function = _transform_activation_function(hidden_activation_function)
    
    return lambda X, y : mlp_soph_fn(X, y, keep_prob_dropout = keep_prob_dropout, embedding_size = embedding_size, 
                                     hidden_layers = hidden_layers, self_normalizing = self_normalizing,
                                     hidden_activation_function=hidden_activation_function,
                                     standard_normal = standard_normal)
    
def cnn(keep_prob_dropout, embedding_size, hidden_layers, pretrained_embeddings_path, trainable_embeddings):
    
    return lambda X, y : cnn_fn(X, y, keep_prob_dropout = keep_prob_dropout, embedding_size = embedding_size, 
                                     hidden_layers = hidden_layers, pretrained_embeddings_path=pretrained_embeddings_path,
                                     trainable_embeddings = trainable_embeddings)
    
def lstm(keep_prob_dropout, embedding_size, hidden_layers, pretrained_embeddings_path, trainable_embeddings):
        
    return lambda X, y : lstm_fn(X, y, keep_prob_dropout = keep_prob_dropout, embedding_size = embedding_size, 
                                     hidden_layers = hidden_layers, pretrained_embeddings_path=pretrained_embeddings_path,
                                     trainable_embeddings = trainable_embeddings)


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
    
    def __init__(self, batch_size = 5, num_epochs = 10, get_model = mlp_base(0.5), threshold = 0.2, learning_rate = 0.1, patience = 5,
                       validation_metric = lambda y1, y2 : f1_score(y1, y2, average = "samples"), 
                       optimize_threshold = True,
                       threshold_window = np.linspace(-0.03, 0.03, num=7),
                       tf_model_path = ".tmp_best_models"):
        """
    
        """
        
        self.get_model = get_model
        
        # enable early stopping on validation set
        self.validation_data_position = None
        
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
        
    def _get_save_model_path(self):
        TMP_FOLDER = self.TF_MODEL_PATH
        if not os.path.exists(TMP_FOLDER):
            os.makedirs(TMP_FOLDER)
        return TMP_FOLDER + "/best-model-" + self.get_model.__name__ + str(datetime.now())
       
    def _calc_num_steps(self, X):
        return int(np.ceil(X.shape[0] / self.batch_size))

    def _compute_validation_score(self, session, X_val_batch, y_val_batch):

        feed_dict = {self.x_tensor: X_val_batch, self.y_tensor: y_val_batch}
        feed_dict.update(self.params_predict)

        if self.validation_metric == "val_loss":
            return session.run(self.loss, feed_dict = feed_dict)

        elif callable(self.validation_metric):
            predictions = session.run(self.predictions, feed_dict = feed_dict)
            y_pred = csr_matrix(predictions > self.threshold)
            if self.optimize_threshold:
                return self.validation_metric(y_val_batch, y_pred), predictions
            else:
                return self.validation_metric(y_val_batch, y_pred)

    def fit(self, X, y):
        self.y = y
        
        val_pos = self.validation_data_position
        
        if val_pos is not None:
            X_train, y_train, X_val, y_val = X[:val_pos, :], y[:val_pos,:], X[val_pos:, :], y[val_pos:, :]
         
            validation_batch_generator = BatchGenerator(X_val, y_val, self.batch_size, False, False)
            validation_predictions = self._calc_num_steps(X_val)
            steps_per_epoch = self._calc_num_steps(X_train)
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
        
        # Name logits Tensor, so that is can be loaded from disk after training
        #logits = tf.identity(logits, name='logits')
        logits = tf.contrib.layers.linear(self.last_layer,
                                                num_outputs=y.shape[1])
        
        # Loss and Optimizer
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y_tensor)
        loss = tf.reduce_sum(losses, axis = 1)
        self.loss = tf.reduce_mean(loss, axis = 0)
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        # prediction
        self.predictions = tf.sigmoid(logits)
        
        session = tf.Session()
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
        for epoch in range(self.num_epochs):
            
            if val_pos is not None and epochs_of_no_improvement == self.patience:
                break
            
            # Loop over all batches
            for batch_i in range(steps_per_epoch):
                X_batch, y_batch = batch_generator._batch_generator()
                feed_dict = {self.x_tensor: X_batch, self.y_tensor: y_batch}
                feed_dict.update(self.params_fit)
                session.run(optimizer, feed_dict = feed_dict)

                # overwrite parameter values for prediction step
                feed_dict.update(self.params_predict)
                loss = session.run(self.loss, feed_dict = feed_dict) 

                # calculate validation loss at end of epoch if early stopping is on
                if batch_i + 1 == steps_per_epoch and val_pos is not None:
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
                            break

                # print progress
                print('Epoch {:>2}/{:>2}, Batch {:>2}/{:>2}, Loss: {:0.4f}, Validation-Score: {:0.4f}, Best Validation-Score: {:0.4f}, Threshold: {:0.2f}'.
                       format(epoch + 1, self.num_epochs, batch_i + 1, steps_per_epoch, loss, avg_validation_score, best_validation_score, self.threshold), end='\r')
                
        self.session = session
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
            feed_dict = {self.x_tensor: X_batch}
            feed_dict.update(self.params_predict)
            pred_batch = session.run(self.predictions, feed_dict = feed_dict)
            prediction[i * self.batch_size:(i+1) * self.batch_size, :] = pred_batch
        
        result = csr_matrix(prediction > self.threshold)
        
        # close the session, since no longer needed
        session.close()
        return result
