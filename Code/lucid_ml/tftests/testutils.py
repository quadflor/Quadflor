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

if __name__ == '__main__':
    tf.test.main()