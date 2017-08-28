import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix

def tf_normalize(X, input_tensor):
    """
    Given data as a sparse matrix X, this function scales an input tensor such that each
    column in the output tensor has mean 0 and variance 1.
    
    >>> X = np.array([[0, 0, 0, 0, 1, 1, 1, 2, 3],
    ...               [1, 1, 0, 2, 3, 4, 5, 1, 0],
    ...               [0, 0 ,4, 5, -1, 4, 2, 1, 0]])
    >>> means = np.mean(X, axis = 0)
    >>> stds = np.std(X, axis = 0)
    >>> scaled_X = (X - means) / stds
    >>> X = csr_matrix(X)
    >>> input_tensor = tf.placeholder(dtype = tf.float32, shape = [3, 9])
    >>> output_tensor = tf_normalize(X, input_tensor)
    >>> sess = tf.Session()
    >>> output = sess.run(output_tensor, feed_dict = {input_tensor : X.toarray()})
    >>> output
    array([[-0.70710683, -0.70710683, -0.70710683, -1.13555002,  0.        ,
            -1.41421354, -0.98058075,  1.41421342,  1.41421354],
           [ 1.41421342,  1.41421342, -0.70710683, -0.16222139,  1.2247448 ,
             0.70710677,  1.37281287, -0.70710689, -0.70710677],
           [-0.70710683, -0.70710683,  1.41421342,  1.29777145, -1.2247448 ,
             0.70710677, -0.39223233, -0.70710689, -0.70710677]], dtype=float32)
    """
    m = X.mean(axis = 0)
        
    # because we are dealing with sparse matrices, we need to compute the variance as
    # E[X^2] - E[X]^2
    X_square = X.power(2)
    m_square = X_square.mean(axis = 0)
    v = m_square - np.power(m, 2)
    s = np.sqrt(v)
    
    
    m = tf.constant(m, dtype = tf.float32) 
    s = tf.constant(s, dtype = tf.float32)
    
    scaled_input = (input_tensor - m) / s
    
    return scaled_input

if __name__ == "__main__":
    import doctest
    doctest.testmod()