import tensorflow as tf
from model import Model
from core.utils.tf_utils import xavier_weight_init, conv2d, \
        max_pool_2x2, weight_variable, bias_variable
from layer import FullyConnected, Dropout, Embedding, Flatten


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

class EmbeddingsInput(Model):
    def get_feed_dict(self, x, d, y=None):
        """
        Return feed dict
        Args:
            x: inputs, tuple with x[0] is ids
                                          x[1] is features
            y: labels
            d: dropout
        """
        feed = {self.ids: x[0],
                self.features: x[1], 
                self.dropout: d}
        if y is not None:
            feed[self.y] = y
        return feed


    def add_placeholder(self):
        """
        Defines self.x, self.y and self.dropout tf.placeholders
        Here, the input is a list of ids of cells as well other information about these cells
        Example [1, 14] and [[8.4, 1234], [-1.2, 1453]] where 
            1 is the id of the cell
            8.4 energy of this cell
            1234 its volume
        """
        self.ids = tf.placeholder(dtype=tf.int32, 
            shape=[None, self.config.max_n_cells], name="ids")
        self.features = tf.placeholder(dtype=tf.float32, 
            shape=[None, self.config.max_n_cells, self.config.n_features], name="features")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None])
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[])

    def add_prediction_op(self):
        """
        Defines self.pred
        """
        # shape = (None, max_n_cells, embedding_size, 1)
        embedding_layer = Embedding(self.config.n_cells, self.config.embedding_size)
        embeddings = embedding_layer(self.ids)
        embeddings = tf.expand_dims(embeddings, axis=3)

        # shape = (None, max_n_cells, 1, n_features)
        features = tf.expand_dims(self.features, axis=2)


        # shape = (None, max_n_cells, embedding_size, n_features)
        feat = tf.matmul(embeddings, features)

        # shape = (None, embedding_size, n_features)
        feat2 = tf.reduce_max(feat, axis=1)

        # shape = (None, embedding_size * n_features)
        f_layer = Flatten()
        f_layer.set_param(input_shape=[None, self.config.embedding_size, self.config.n_features])
        feat3 = f_layer(feat2)

        # fully connected before pred
        fc_layer = FullyConnected(self.config.output_size)
        fc_layer.set_param(input_shape=[None, self.config.embedding_size*self.config.n_features])
        self.pred = fc_layer(feat3)





