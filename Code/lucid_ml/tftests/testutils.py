from utils.tf_utils import *
import tensorflow as tf

class SequenceTests(tf.test.TestCase):

    def testSequenceLength(self):
        with self.test_session():
            sequences = [[3, 4, 1, 3, 6, 0, 0, 0],
                         [3, 4, 1, 3, 6, 4, 0, 0],
                         [3, 4, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 0, 0, 0, 0],
                         [3, 4, 1, 3, 6, 1, 1, 1]]
            lengths = sequence_length(sequences)
            self.assertAllEqual(lengths.eval(), [5,6,3,0,1,8])
            
            
class AggregationTests(tf.test.TestCase):

    def testAverage(self):
        with self.test_session():
            # output dimensions = [3, 4, 2]
            sequences = [[3, 4, 1, 3],
                         [0, 0, 0, 0],
                         [3, 0, 0, 0],
                         [1, 2, 0, 0]]
            outputs = [[[1., 1.], [1., 1.], [2., 2.5], [3., 4.]],
                       [[1., 2.], [3., 4.], [0., 0.], [0., 0.]],
                       [[5., 6.], [10., 10.,], [10., 10.], [10., 10.]],
                       [[10., 12.], [0., 0.,],[0., 0.,],[0., 0.,]]]
            outputs = tf.constant(outputs, dtype = tf.float32)
            expected_result = [[1.75, 2.125],
                               [0., 0.],
                               [5., 6.],
                               [5., 6.]]
            lengths = sequence_length(sequences)
            result = average_outputs(outputs, lengths)
            self.assertAllEqual(result.eval(), expected_result)
            
            
class DynamicMaxPoolingTests(tf.test.TestCase):
    
    def testPoolingMask(self):
        pass

    def testDynamicMaxPooling(self):
        with self.test_session():
            
            # add dimension to detector stage
            window_size = 1
            num_filters = 3
            max_length = 3
            map_size = max_length - window_size + 1
            #detector : tensor of shape [batch_size, map_size, 1, num_filters]
            detector = [[[1., 1., 3.], [2., 2.5, 1.5], [3., 4., 2.]],
                       [[1., 2., 0.5], [0., 0., 1.], [0., 0., 1.]],
                       [[5., 6., 20.], [10., 10., 9.5], [10., 10., 11.]],
                       [[10., 12., 0.], [4., 3., 1.],[2., 2., 7.]]]
            detector = tf.expand_dims(detector, axis = 2)
            
            seq_length = tf.constant([0, 1, 2, 3], dtype = tf.float32)
            
            # p = 1
            result = dynamic_max_pooling(detector, seq_length, max_length, num_filters, window_size, dynamic_max_pooling_p = 1)
            expected_result = [[0., 0., 0.],
                               [1., 2., 0.5],
                               [10., 10., 20.],
                               [10., 12., 7.]]
            self.assertAllEqual(result.eval(), expected_result)
            
            # full sequence length, p = 2, so we check if we can handle the case where text-length divided by number of chunks is not an integer
            seq_length = tf.constant([3, 3, 3, 3], dtype = tf.float32)
            result = dynamic_max_pooling(detector, seq_length, max_length, num_filters, window_size, dynamic_max_pooling_p = 2)
            expected_result = [[2., 2.5, 3., 3., 4., 2.],
                               [1., 2., 1., 0., 0., 1.],
                               [10., 10., 20., 10., 10., 11.],
                               [10., 12., 1., 2., 2., 7. ]]
            self.assertAllEqual(result.eval(), expected_result)
            
            # full sequence length, p = 3, which should collapse the sequence dimension of the detector outputs in this case
            seq_length = tf.constant([3, 3, 3, 3], dtype = tf.float32)
            result = dynamic_max_pooling(detector, seq_length, max_length, num_filters, window_size, dynamic_max_pooling_p = 3)
            expected_result = [[1., 1., 3., 2., 2.5, 1.5, 3., 4., 2.],
                               [1., 2., 0.5, 0., 0., 1., 0., 0., 1.],
                               [5., 6., 20., 10., 10., 9.5, 10., 10., 11.],
                               [10., 12., 0., 4., 3., 1., 2., 2., 7.]]
            self.assertAllEqual(result.eval(), expected_result)

if __name__ == '__main__':
    tf.test.main()