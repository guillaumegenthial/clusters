import tensorflow as tf
from model import Model
from core.utils.tf_utils import xavier_weight_init, conv2d, \
        max_pool_2x2, weight_variable, bias_variable
from layer import FullyConnected, Dropout


class FlatInput(Model):
    def __init__(self, config, input_size, layers=None):
        Model.__init__(self, config, layers)
        self.config.input_size = input_size

    def add_placeholder(self):
        """
        Defines self.x and self.y, tf.placeholders
        """
        self.x_shape = [None, self.config.input_size]
        self.x = tf.placeholder(tf.float32, shape=self.x_shape)
        self.y = tf.placeholder(tf.int32, shape=[None])
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[])

class SquareInput(Model):
    def __init__(self, config, n_phi, n_eta, n_features, layers=None):
        Model.__init__(self, config, layers)
        self.config.n_phi = n_phi
        self.config.n_eta = n_eta
        self.config.n_features = n_features

    def add_placeholder(self):
        n_phi = self.config.n_phi
        n_eta = self.config.n_eta
        n_features = self.config.n_features

        self.x_shape = [None, n_phi, n_eta, n_features]
        self.x = tf.placeholder(tf.float32, shape=self.x_shape)
        self.y = tf.placeholder(tf.int32, shape=[None])
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[])


