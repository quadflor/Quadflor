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
    
    #make sure not to divide by zero when scaling
    s[s == 0] = 1
    
    m = tf.constant(m, dtype = tf.float32) 
    s = tf.constant(s, dtype = tf.float32)
    
    scaled_input = (input_tensor - m) / s
    
    return scaled_input

def sequence_length(sequence):
    """
    Takes as input a tensor of dimensions [batch_size, max_len] which encodes some sequence of maximum length max_len as
    a sequence of positive ids. For positions beyond the length of the actual sequence, the id is assumed to be zero (i.e., zero is used for padding).
    
    This function returns a one-dimensional tensor of size [batch_size], where each entry denotes the length of the corresponding sequence.
    """
    used = tf.sign(sequence)
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def dynamic_max_pooling(detector, seq_length, max_length, num_filters, window_size, dynamic_max_pooling_p = 1):
    """
    Performs dynamic max pooling as in [XML-CNN]. That is, it splits the outputs of the detector stage into 'p' chunks, performs
    max-pooling on each chunk and concatenates the outputs. The function assumes the CNN from which the detector stage results to use 'VALID'
    padding as well as strides = [1, 1, 1, 1].
    
    If the length of the sequence can not be divided into evenly sized chunks, we make the last chunk contain the remainder of the text.
    
    References
    ----------
    [XML-CNN] J Liu, WC Chang, Y Wu, Y Yang 
            "Deep Learning for Extreme Multi-label Text Classification" 
            Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval
    
    Parameters
    ----------
    detector : tensor of shape [batch_size, map_size, 1, num_filters]
        The output from the 'detector' stage of a CNN with a configuration (number of filters, window_size, ...) such that it results in
        a tensor with the according feature map size and number of filters.
    seq_length : tensor of shape [batch_size] containing the length of the text.
    max_length : maximum length of the sequence
    num_filters : number of filters used in the CNN
    window_size : window size used in the CNN
    dynamic_max_pooling_p : the number of chunks 'p' to split the outputs of the detector stage into.
    
    Returns
    -------
    Tensor of shape [batch_size, p * num_filters]
        The concatenated outputs of the max-pooling operations over the 'p' chunks.
    """
    # assumptions of CNN from which the detector outputs result
    stride = [1,1,1,1]
    detector_output_length = seq_length - window_size + 1
    
    # dynamic max-pooling: extract maximum for each chunk
    chunks_size = tf.ceil(tf.divide(detector_output_length, dynamic_max_pooling_p))
    chunk_poolings = []  
    for i in range(dynamic_max_pooling_p):
        
        # make sure we don't get out of bounds at end of sequence
        cur_chunk_size = chunks_size if i != dynamic_max_pooling_p - 1 or dynamic_max_pooling_p == 1 else detector_output_length - i * chunks_size
        
        # create a mask for the entire sequence where only those are selected which are in the current chunk
        start_indices = i * chunks_size
        end_indices = i * chunks_size + cur_chunk_size
        
        neg_mask_start = 1 - tf.cast(tf.sequence_mask(start_indices, maxlen = max_length - window_size + 1), tf.float32)
        mask_end = tf.cast(tf.sequence_mask(end_indices, maxlen = max_length - window_size + 1), tf.float32)
        final_mask = tf.multiply(neg_mask_start, mask_end)
        final_mask = tf.expand_dims(final_mask, axis = 2)
        final_mask = tf.expand_dims(final_mask, axis = 3)
        
        extracted_chunk = tf.multiply(final_mask, detector)
        pooling = tf.nn.max_pool(extracted_chunk,
                                ksize = [1, max_length - window_size + 1, 1, 1],
                                strides = stride,
                                padding = "VALID")
        pooling = tf.reshape(pooling, [-1, num_filters])
        chunk_poolings.append(pooling)
    
    concatenated_pooled_chunks = tf.concat(chunk_poolings, 1)
    return concatenated_pooled_chunks

def average_outputs(outputs, seq_length):
    """
    Given the padded outputs of an RNN and the actual length of the sequence, this function computes the average
    over all (non-padded) outputs. In the special case where the length is 0, the function returns 0.
    
    Parameters
    ----------
    outputs : tensor of shape [batch_size, max_length, output_dimensions]
        The output from an RNN with hidden representation size 'output_dimensions'.
    seq_length : tensor of shape [batch_size] containing the number of valid outputs in 'outputs'.
    
    Returns
    -------
    Tensor of shape [batch_size, output_dimensions]
        The average over all outputs in the sequence.
    """
    # average over outputs at all time steps
    seq_mask = tf.cast(tf.sequence_mask(seq_length, maxlen = outputs.get_shape().as_list()[1]), tf.float32)
    seq_mask = tf.expand_dims(seq_mask, axis = 2)
    outputs = outputs * seq_mask
    output_state = tf.reduce_sum(outputs, axis = 1)
    seq_length_reshaped = tf.cast(tf.reshape(seq_length, [-1, 1]), tf.float32)
    minimum_length = tf.ones_like(seq_length_reshaped, dtype=tf.float32)
    output_state = tf.div(output_state, tf.maximum(seq_length_reshaped, minimum_length))
    return output_state

if __name__ == "__main__":
    import doctest
    doctest.testmod()