import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from scipy.sparse.csr import csr_matrix
from sklearn.base import BaseEstimator
tf.logging.set_verbosity(tf.logging.INFO)

def mlp_soph_fn(features, targets, mode, params):
    """Model function for MLP-Soph."""

    targets = tf.to_float(targets)
    print("Creating the model")
    # Connect the first hidden layer to input layer
    # (features) with relu activation
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
        predictions=predictions,
        loss=loss,
        train_op=train_op)

class BatchGenerator:
    
    def __init__(self, X, y, batch_size, shuffle, predict):
        self.X = X
        self.y = y
        self.number_of_batches = np.ceil(X.shape[0] / batch_size)
        self.counter = 0
        self.sample_index = np.arange(X.shape[0])
        self.batch_size = batch_size
        self.predict = predict
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
            return tf.constant(X_batch), tf.constant(y_batch)
        else:
            return tf.constant(X_batch)

def mlp_soph():
    return mlp_soph_fn, {"learning_rate": 0.01}

class MultiLabelSKFlow(BaseEstimator):
    """
    This is a wrapper class for tf.contrib.learn.Estimators, so it adheres to the fit/predict naming conventions of sk-learn.
    It already handles mini-batching, whose behavior can be controlled by providing the respective parameters to the init function.
    
    The concrete TensorFlow model to execute can be specified in terms of the 'get_model' function.
    This function in turn has to return a 'model_fn' function and a 'params' dictionary.
    These arguments will passed to the tf.contrib.learn.Estimator class and have thus to conform the formats
    described in 'https://www.tensorflow.org/extend/estimators'.
    On top of that, 'model_fn' has to assume the 'features' and 'targets' parameters to be of the Tensor class.
    """
    
    def __init__(self, batch_size = 5, num_epochs = 10, get_model = mlp_soph):
        """
    
        """
        
        # used by this class
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # used by the tensorflow model function
        model_fn, model_params = get_model()
        
        self._estimator = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params)
        
        
    def fit(self, X, y):
        
        self.X = X
        self.y = y
        batch_generator = BatchGenerator(X, y, self.batch_size, True, False)
        steps_per_epoch = int(np.ceil(X.shape[0] / self.batch_size))
        self._estimator.fit(input_fn=lambda : batch_generator._batch_generator(), steps=self.num_epochs * steps_per_epoch)
        
        
    def predict(self, X):
        
        # TODO: implement batch prediction
        prediction_steps = int(np.ceil(X.shape[0] / self.batch_size))
        
        prediction = np.zeros((X.shape[0], self.y.shape[1]))
        for i in range(prediction_steps):
            batch_generator = BatchGenerator(X, None, self.batch_size, False, True)
            pred_generator = self._estimator.predict(input_fn=lambda : batch_generator._batch_generator())
            pred_batch = np.array([single_prediction for single_prediction in pred_generator])
            print(pred_batch.shape)
            prediction[i * self.batch_size:(i+1) * self.batch_size] = pred_batch
        
        threshold = 0.2
        prediction[prediction > threshold] = 1
        prediction[prediction <= threshold] = 0
        return csr_matrix(prediction)
        
        