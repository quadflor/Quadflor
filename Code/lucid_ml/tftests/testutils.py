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

if __name__ == '__main__':
    tf.test.main()