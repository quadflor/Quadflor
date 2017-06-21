import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from scipy.sparse.csr import csr_matrix
from sklearn.base import BaseEstimator
import math

#tf.logging.set_verbosity(tf.logging.INFO)

#===============================================================================
# def cnn_fn(features, targets, mode, params, num_classes):
#     """Model function for CNN."""
#     # convert sparse tensors to dense
#     
#     EMBEDDING_SIZE = 50 
#     
#     W = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0),
#                     name="W")
#                     self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
#                     self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
# 
#     if mode != tf.contrib.learn.ModeKeys.INFER:
#         targets = tf.sparse_reorder(targets)
#         targets = tf.sparse_tensor_to_dense(targets)
#         targets = tf.to_float(targets)
#         targets = tf.reshape(targets, [num_samples, num_classes])
# 
#     # Connect the first hidden layer to input layer
#     # (features) with relu activation and add dropout
#     hidden_layer = tf.contrib.layers.relu(features, 1000)
#     hidden_dropout = tf.nn.dropout(hidden_layer, params["dropout"])
#     
#     
#     # Connect the output layer to second hidden layer (no activation fn)
#     
#     output_layer = tf.contrib.layers.linear(hidden_dropout,
#                                                  num_outputs=num_classes)
#     
#     output_layer = tf.to_float(output_layer)
# 
#     # Reshape output layer to 1-dim Tensor to return predictions
#     #predictions = tf.reshape(output_layer, [-1])
#     predictions = tf.contrib.layers.fully_connected(inputs=hidden_dropout,
#                                                  num_outputs=num_classes,
#                                                  activation_fn=tf.sigmoid)
#     
#     # Calculate loss using mean squared error
#     if mode != tf.contrib.learn.ModeKeys.INFER:
#         losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = targets, logits = output_layer)
#         loss = tf.reduce_sum(losses)
#     else:
#         loss = tf.constant(0.)
#     
#     if mode != tf.contrib.learn.ModeKeys.INFER:
#         train_op = tf.contrib.layers.optimize_loss(
#             loss=loss,
#             global_step=tf.contrib.framework.get_global_step(),
#             learning_rate=params["learning_rate"],
#             optimizer="Adam")
#     else:
#         train_op = None
#     return model_fn_lib.ModelFnOps(
#         mode=mode,
#         predictions=predictions,
#         loss=loss,
#         train_op=train_op)
#===============================================================================
def mlp_soph_fn(X, y, dropout = 0.5):
    """Model function for MLP-Soph."""
    # convert sparse tensors to dense
    x_tensor = tf.placeholder(tf.float32, shape=(None, X.shape[1]), name = "x")
    y_tensor = tf.placeholder(tf.float32, shape=(None, y.shape[1]), name = "y")
    dropout_tensor = tf.placeholder(tf.float32, name = "dropout")
    
    params_fit = {dropout_tensor : dropout}
    params_predict = {dropout_tensor : 1}
    
    # Connect the first hidden layer to input layer
    # (features) with relu activation and add dropout
    hidden_layer = tf.contrib.layers.relu(x_tensor, 1000)
    hidden_dropout = tf.nn.dropout(hidden_layer, dropout_tensor)
    
    return x_tensor, y_tensor, hidden_dropout, params_fit, params_predict

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
        X_batch = self.X[batch_index, :].toarray()
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


def mlp_soph(dropout):
    return lambda X, y : mlp_soph_fn(X, y, dropout = dropout)


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
    
    def __init__(self, batch_size = 5, num_epochs = 10, get_model = mlp_soph(0.5), threshold = 0.2, learning_rate = 0.1):
        """
    
        """
        
        self.get_model = get_model
        
        # enable early stopping on validation set
        self.validation_data_position = None
        
        # used by this class
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.threshold = threshold
        if learning_rate is None:
            self.learning_rate = self.batch_size / 512 * 0.01
        else:
            self.learning_rate = learning_rate
        
        # path to save the tensorflow model to
        self._save_model_path = './multi-label-model'
       
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
        for epoch in range(self.num_epochs):
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

                # print progress
                print('Epoch {:>2}/{}, Batch {}/{}, Loss: {}, Validation-Loss: {}'.format(epoch + 1, self.num_epochs,
                                                                                          batch_i + 1, steps_per_epoch, 
                                                                                          loss, avg_validation_loss), end='\r')
                
        self.session = session
        print('')
        
        # Save model for prediction step
        #saver = tf.train.Saver()
        #saver.save(sess, self._save_model_path)
    
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
        # Load model
        #loader = tf.train.import_meta_graph(self._save_model_path + '.meta')
        #loader.restore(sess, self._save_model_path)

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
        return result
