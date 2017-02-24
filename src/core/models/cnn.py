import tensorflow as tf
from model import Model
from core.utils.tf_utils import xavier_weight_init, conv2d, \
        max_pool_2x2, weight_variable, bias_variable

class Conv2d(Model):
    def __init__(self, config, n_phi, n_eta, n_features):
        Model.__init__(self, config)
        self.config.n_phi = n_phi
        self.config.n_eta = n_eta
        self.config.n_features = n_features

    def add_placeholder(self):
        n_phi = self.config.n_phi
        n_eta = self.config.n_eta
        n_features = self.config.n_features

        self.x = tf.placeholder(tf.float32, shape=[None, n_phi, n_eta, n_features])
        self.y = tf.placeholder(tf.int32, shape=[None])
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[])

    def add_prediction_op(self):
        """
        Defines self.pred
        """
        x = self.x
        k = 10

        W_conv = weight_variable('W_conv', [5, 5, self.config.n_features, k])
        b_conv = bias_variable('b_conv', [10])
        h_conv  = tf.nn.relu(conv2d(x, W_conv) + b_conv)

        h_pool = max_pool_2x2(h_conv)
        h_pool_shape = [(self.config.n_phi + 1)/2, (self.config.n_eta + 1) /2]
        h_pool_flat = tf.reshape(h_pool, [-1, h_pool_shape[0]*h_pool_shape[1]*k])

        W_fc1 = weight_variable('W_fc1', [h_pool_shape[0]*h_pool_shape[1] * k, 10])
        b_fc1 = bias_variable('b_fc1', [10])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

        h_fc_drop = tf.nn.dropout(h_fc1, self.dropout)

        W_fc2 = weight_variable('W_fc2', [10, self.config.output_size])
        b_fc2 = bias_variable('b_fc2', [self.config.output_size])

        self.pred = tf.matmul(h_fc_drop, W_fc2) + b_fc2


