import tensorflow as tf
from model import Model
from core.utils.tf_utils import xavier_weight_init, conv2d, \
        max_pool_2x2, weight_variable, bias_variable

class Regression(Model):
    def __init__(self, config, input_size):
        Model.__init__(self, config)
        self.config.input_size = input_size

    def add_placeholder(self):
        """
        Defines self.x and self.y, tf.placeholders
        """
        self.x = tf.placeholder(tf.float32, shape=[None, self.config.input_size])
        self.y = tf.placeholder(tf.int32, shape=[None, self.config.output_size])
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[])

    def get_x(self):
        return self.x

    def add_prediction_op(self):
        """
        Defines self.pred
        """
        x = self.get_x()
        x = tf.nn.dropout(x, self.dropout)
        W = weight_variable('W', [self.config.input_size, self.config.output_size])
        b = bias_variable('bias', [self.config.output_size])
        self.pred = tf.matmul(x, W) + b


class MultiRegression(Regression):
    def add_prediction_op(self):
        """
        Defines self.pred
        """
        hidden_sizes = [self.config.input_size] + self.config.hidden_sizes + [self.config.output_size]
        temp = self.get_x()

        for i in range(len(hidden_sizes) - 1):
            # temp = tf.nn.dropout(temp, self.dropout)
            fan_in = hidden_sizes[i]
            fan_out = hidden_sizes[i+1]
            W = weight_variable('W_{}'.format(i), [fan_in, fan_out])
            b = bias_variable('bias_{}'.format(i), [fan_out])
            temp = tf.matmul(temp, W) + b
            if i + 1 != len(hidden_sizes) - 1:
                temp = tf.nn.relu(temp)

        self.pred = temp

class RawRegression(MultiRegression):
    def __init__(self, config, n_phi, n_eta, n_features):
        MultiRegression.__init__(self, config, n_phi*n_eta*n_features)
        self.config.n_phi = n_phi
        self.config.n_eta = n_eta
        self.config.n_features = n_features

    def add_placeholder(self):
        n_phi = self.config.n_phi
        n_eta = self.config.n_eta
        n_features = self.config.n_features

        self.x = tf.placeholder(tf.float32, shape=[None, n_phi, n_eta, n_features])
        self.y = tf.placeholder(tf.int32, shape=[None, self.config.output_size])
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[])

    def get_x(self):
        n_phi = self.config.n_phi
        n_eta = self.config.n_eta
        n_features = self.config.n_features

        shape_flat = n_phi * n_eta * n_features
        x_flat = tf.reshape(self.x, [-1, shape_flat])

        return x_flat



