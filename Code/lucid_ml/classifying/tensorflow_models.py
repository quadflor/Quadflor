import numpy as np
import tensorflow as tf
from scipy.sparse.csr import csr_matrix
import scipy.sparse as sps
from sklearn.base import BaseEstimator
import math, numbers
from tensorflow.python.framework import ops, tensor_shape,  tensor_util
from tensorflow.python.ops import math_ops, random_ops, array_ops
from tensorflow.python.layers import utils

#tf.logging.set_verbosity(tf.logging.INFO)

def cnn_fn(X, y, keep_prob_dropout = 0.5, embedding_size = 30, hidden_layers = [1000]):
    """Model function for CNN."""
     
    # set vocab size to the largest word identifier that exists in the training data + 1
    vocab_size = int(np.max(X)) + 1
    sequence_length = X.shape[1]
    x_tensor = tf.placeholder(tf.int32, shape=(None, sequence_length), name = "x")
    y_tensor = tf.placeholder(tf.float32, shape=(None, y.shape[1]), name = "y")
    dropout_tensor = tf.placeholder(tf.float32, name = "dropout")
    
    params_fit = {dropout_tensor : 1 - keep_prob_dropout}
    params_predict = {dropout_tensor : 1}
    
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        lookup_table = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                        name="W")
        embedded_words = tf.nn.embedding_lookup(lookup_table, x_tensor)
        
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
    
    return x_tensor, y_tensor, hidden_layer, params_fit, params_predict
def mlp_base_fn(X, y, keep_prob_dropout = 0.5):
    """Model function for MLP-Soph."""
    # convert sparse tensors to dense
    x_tensor = tf.placeholder(tf.float32, shape=(None, X.shape[1]), name = "x")
    y_tensor = tf.placeholder(tf.float32, shape=(None, y.shape[1]), name = "y")
    dropout_tensor = tf.placeholder(tf.float32, name = "dropout")
    
    params_fit = {dropout_tensor : keep_prob_dropout}
    params_predict = {dropout_tensor : 1}
    
    # Connect the first hidden layer to input layer
    # (features) with relu activation and add dropout
    hidden_layer = tf.contrib.layers.relu(x_tensor, 1000)
    hidden_dropout = tf.nn.dropout(hidden_layer, dropout_tensor)
    
    return x_tensor, y_tensor, hidden_dropout, params_fit, params_predict

# https://github.com/bioinf-jku/SNNs/blob/master/selu.py
def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
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

def mlp_soph_fn(X, y, keep_prob_dropout = 0.5, embedding_size = 30, hidden_layers = [1000], self_normalizing = True):
    """Model function for MLP-Soph."""
    # convert sparse tensors to dense
    x_tensor = tf.placeholder(tf.float32, shape=(None, X.shape[1]), name = "x")
    y_tensor = tf.placeholder(tf.float32, shape=(None, y.shape[1]), name = "y")
    dropout_tensor = tf.placeholder(tf.float32, name = "dropout")
    
    params_fit = {dropout_tensor : 1 - keep_prob_dropout}
    params_predict = {dropout_tensor : 1}
    
    # apply a look-up as described by the fastText paper
    if embedding_size > 0:
        lookup_table = tf.Variable(tf.truncated_normal([X.shape[1], embedding_size], mean=0.0, stddev=0.1))
        embedding_layer = tf.matmul(x_tensor, lookup_table)
    else:
        embedding_layer = x_tensor
    
    # Connect the embedding layer to the hidden layers
    # (features) with relu activation and add dropout everywhere
    hidden_layer = embedding_layer
    for hidden_units in hidden_layers:
        if not self_normalizing:
            hidden_layer = tf.contrib.layers.relu(hidden_layer, hidden_units)
            hidden_layer = tf.nn.dropout(hidden_layer, dropout_tensor)
        else:
            hidden_layer = tf.contrib.layers.fully_connected(hidden_layer, hidden_units,
                                                      activation_fn=None,
                                                      weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
            hidden_layer = selu(hidden_layer)
            hidden_layer = dropout_selu(hidden_layer, dropout_tensor)
    
    return x_tensor, y_tensor, hidden_layer, params_fit, params_predict

def mlp_base(keep_prob_dropout):
    return lambda X, y : mlp_base_fn(X, y, keep_prob_dropout = keep_prob_dropout)

def mlp_soph(keep_prob_dropout, embedding_size, hidden_layers, self_normalizing):
    return lambda X, y : mlp_soph_fn(X, y, keep_prob_dropout = keep_prob_dropout, embedding_size = embedding_size, 
                                     hidden_layers = hidden_layers, self_normalizing = self_normalizing)
    
def cnn(keep_prob_dropout, embedding_size, hidden_layers):
    return lambda X, y : cnn_fn(X, y, keep_prob_dropout = keep_prob_dropout, embedding_size = embedding_size, 
                                     hidden_layers = hidden_layers)


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



#===============================================================================
# def lstm(lr, dropout):
#     if lr is None:
#         # set lr to default option
#         lr = 0.1
#     return lambda : (cnn_fn, {"learning_rate": lr, "dropout" : dropout})
#===============================================================================

#===============================================================================
# def _sparse_to_sparse(X):
#     non_zero_rows, non_zero_cols = X.nonzero()
#     indices = np.array([[r, c] for r, c in zip(non_zero_rows, non_zero_cols)])
#     values = np.array([X[r,c] for r, c in zip(non_zero_rows, non_zero_cols)])
#      
#     shape = list(X.shape)
#     input_tensor = tf.SparseTensor(indices=indices,
#                                    values=values,
#                                    dense_shape=shape)
#     return input_tensor
#===============================================================================



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
    
    def __init__(self, batch_size = 5, num_epochs = 10, get_model = mlp_base(0.5), threshold = 0.2, learning_rate = 0.1, tolerance = 5):
        """
    
        """
        
        self.get_model = get_model
        
        # enable early stopping on validation set
        self.validation_data_position = None
        
        # used by this class
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.threshold = threshold
        if learning_rate is None:
            self.learning_rate = self.batch_size / 512 * 0.01
        else:
            self.learning_rate = learning_rate
        
        # path to save the tensorflow model to
        self._save_model_path = './best-model'
       
    def _calc_num_steps(self, X):
        return int(np.ceil(X.shape[0] / self.batch_size))
       
    def fit(self, X, y):
        self.y = y
        
        val_pos = self.validation_data_position
        
        if val_pos is not None:
            X_train, y_train, X_val, y_val = X[:val_pos, :], y[:val_pos,:], X[val_pos:, :], y[val_pos:, :]
         
            validation_batch_generator = BatchGenerator(X_val, y_val, X_val.shape[0], False, False)
            validation_predictions = self._calc_num_steps(X_val)
            steps_per_epoch = self._calc_num_steps(X_train)
        else:
            steps_per_epoch = self._calc_num_steps(X)
            X_train = X
            y_train = y
         
        # Remove previous weights, bias, inputs, etc..
        tf.reset_default_graph()
        
        # Inputs
        
        # get_model has to return a 
        self.x_tensor, self.y_tensor, self.last_layer, self.params_fit, self.params_predict = self.get_model(X, y)
        
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
        
        batch_generator = BatchGenerator(X_train, y_train, self.batch_size, True, False)
        # Training cycle
        avg_validation_loss = math.inf
        best_validation_loss = math.inf
        epochs_of_no_improvement = 0
        for epoch in range(self.num_epochs):
            
            if val_pos is not None and epochs_of_no_improvement == self.tolerance:
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
                    validation_losses = []
                    weights = []
                    for _ in range(validation_predictions):
                        X_val_batch, y_val_batch = validation_batch_generator._batch_generator()
                        weights.append(X_val_batch.shape[0])
                        feed_dict = {self.x_tensor: X_val_batch, self.y_tensor: y_val_batch}
                        feed_dict.update(self.params_predict)
                        validation_losses.append(session.run(self.loss, feed_dict = feed_dict))
                    avg_validation_loss = np.average(np.array(validation_losses), weights = np.array(weights))

                    if avg_validation_loss < best_validation_loss:
                        # save model
                        # Save model for prediction step
                        best_validation_loss = avg_validation_loss
                        saver = tf.train.Saver()
                        saver.save(session, self._save_model_path)
                        epochs_of_no_improvement = 0
                    else:
                        epochs_of_no_improvement += 1
                        if epochs_of_no_improvement > self.tolerance:
                            print("No improvement in validation loss for", self.tolerance, "epochs. Stopping early.")
                            break

                # print progress
                print('Epoch {:>2}/{:>2}, Batch {:>2}/{:>2}, Loss: {:0.4f}, Validation-Loss: {:0.4f}, Best Validation-Loss: {:0.4f}'.format(epoch + 1, self.num_epochs,
                                                                                          batch_i + 1, steps_per_epoch, 
                                                                                          loss, avg_validation_loss, best_validation_loss), end='\r')
                
        self.session = session
        print('')
    
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
