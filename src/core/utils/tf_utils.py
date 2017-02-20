import numpy as np
import tensorflow as tf


def xavier_weight_init():
    """Returns function that creates random tensor.

    The specified function will take in a shape (tuple or 1-d array) and
    returns a random tensor of the specified shape drawn from the
    Xavier initialization distribution.

    Hint: You might find tf.random_uniform useful.
    # look at tf.contrib
    """
    def _xavier_initializer(shape, **kwargs):
        """Defines an initializer for the Xavier distribution.
        Specifically, the output should be sampled uniformly from [-epsilon, epsilon] where
            epsilon = sqrt(6) / <sum of the sizes of shape's dimensions>
        e.g., if shape = (2, 3), epsilon = sqrt(6 / (2 + 3))

        This function will be used as a variable initializer.

        Args:
            shape: Tuple or 1-d array that species the dimensions of the requested tensor.
        Returns:
            out: tf.Tensor of specified shape sampled from the Xavier distribution.
        """
        ### YOUR CODE HERE
        epsilon = np.sqrt(6) / np.sqrt(sum(shape))
        out = tf.random_uniform(shape, minval=-epsilon, maxval=epsilon)
        ### END YOUR CODE
        return out
    # Returns defined initializer function.
    return _xavier_initializer


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(name, shape, initializer=xavier_weight_init()):
  initial = initializer(shape)
  return tf.Variable(initial, name=name)

def bias_variable(name, shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)