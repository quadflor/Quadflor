import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from scipy.sparse.csr import csr_matrix
import scipy.sparse as sps
from pandas.core.sparse.array import _sparray_doc_kwargs
tf.logging.set_verbosity(tf.logging.INFO)

def mlp_soph_fn(features, targets, mode, params):
    """Model function for MLP-Soph."""

    targets = tf.to_float(targets)
    print("Creating the model")
    # Connect the first hidden layer to input layer
    # (features) with relu activation
    features = features["bow"]
    first_hidden_layer = tf.contrib.layers.relu(features, 10)
    
    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 10)
    
    # Connect the output layer to second hidden layer (no activation fn)
    
    num_classes = targets.get_shape()[1].value
    output_layer = tf.contrib.layers.linear(second_hidden_layer,
                                                 num_outputs=num_classes)
    
    # cast to float32
    output_layer = tf.to_float(output_layer)

    # Reshape output layer to 1-dim Tensor to return predictions
    #predictions = tf.reshape(output_layer, [-1])
    predictions = tf.contrib.layers.fully_connected(inputs=second_hidden_layer,
                                                 num_outputs=num_classes,
                                                 activation_fn=tf.sigmoid)
    predictions_dict = {"labels": predictions}
    
    # Calculate loss using mean squared error
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = targets, logits = output_layer)
    loss = tf.reduce_sum(losses)
    
    # Calculate root mean squared error as additional eval metric
    #===========================================================================
    # eval_metric_ops = {
    #     "rmse":
    #         tf.metrics.root_mean_squared_error(
    #             tf.cast(targets, tf.float64), predictions)
    # }
    #===========================================================================
    
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer="SGD")
    print("Done creating the model I guess")
    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op)

def _batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        y_batch = y[batch_index].toarray()
        counter += 1
        yield tf.constant(X_batch), tf.constant(y_batch)
        if counter == number_of_batches:
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

class MLP_Soph():
    
    def _sparse_to_sparse(self, X):
        non_zero_rows, non_zero_cols = X.nonzero()
        indices = np.array([[r, c] for r, c in zip(non_zero_rows, non_zero_cols)])
        values = np.array([X[r,c] for r, c in zip(non_zero_rows, non_zero_cols)])
        
        shape = list(self.X.shape)
        input_tensor = tf.SparseTensor(indices=indices,
                            values=values,
                            dense_shape=shape)
        return input_tensor
    
    def _my_input_fn_fit(self, X, y, predict = False):
        if sps.issparse(X):
            #input_tensor = self._sparse_to_sparse(self.X)
            input_tensor = tf.constant(X.todense())
        else:
            input_tensor = tf.constant(X)
        
        if not predict:
            if sps.issparse(self.y):
                #label_tensor = self._sparse_to_sparse(self.y)
                label_tensor = tf.constant(y.todense())
            else:
                label_tensor = tf.constant(y)
        
        
        feature_cols = {"bow" : input_tensor}
        if not predict:
            return feature_cols, label_tensor
        else:
            return feature_cols
    
    def __init__(self):
        
        model_params = {"learning_rate": 0.1}
        
        model_fn = mlp_soph_fn
        self._estimator = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params)
        
        
    def fit(self, X, y):
        
        self.X = X
        self.y = y
        self._estimator.fit(input_fn=lambda : self._my_input_fn_fit(X, y, predict=False), steps=2000)
        
        
    def predict(self, X):
        prediction = self._estimator.predict(input_fn=lambda : self._my_input_fn_fit(X, None, predict=True), as_iterable = False)["labels"]
        
        threshold = 0.2
        prediction[prediction > threshold] = 1
        prediction[prediction <= threshold] = 0
        return csr_matrix(prediction)
        
        