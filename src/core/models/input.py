import os
import tensorflow as tf
import numpy as np
from model import Model
from core.utils.tf import xavier_weight_init, conv2d, \
        max_pool_2x2, weight_variable, bias_variable
from layer import FullyConnected, Dropout, Embedding, Flatten
from tensorflow.contrib.tensorboard.plugins import projector


class FlatInput(Model):
    def __init__(self, config, input_size, layers=None):
        Model.__init__(self, config, layers)
        self.config.input_size = input_size

    def add_placeholder(self):
        """
        Defines self.x and self.y, tf.placeholders
        """
        self.x = tf.placeholder(tf.float32, 
            shape=[None, self.config.input_size])
        self.y = tf.placeholder(tf.int32, shape=[None])
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[])
        self.lr = tf.placeholder(dtype=tf.float32, shape=[])

        self.nodes = {"x": self.x}
        self.shapes = {"x": [None, self.config.input_size]}

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

        self.x = tf.placeholder(tf.float32, 
            shape=[None, n_phi, n_eta, n_features])
        self.y = tf.placeholder(tf.int32, shape=[None])
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[])
        self.lr = tf.placeholder(dtype=tf.float32, shape=[])

        self.nodes = {"x": self.x}
        self.shapes = {"x": [None, n_phi, n_eta, n_features]}


class IdInput(Model):
    def get_feed_dict(self, x, d, lr=None, y=None, m=None):
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
        if lr is not None:
            feed[self.lr] = lr
        return feed


    def add_placeholder(self):
        """
        Defines self.x, self.y and self.dropout tf.placeholders
        Here, the input is a list of ids of cells as well other information about these cells
        Example [1, 14] and [[8.4, 1234], [-1.2, 1453]] where 
        
            8.4 energy of this cell
            1234 its volume
        """
        self.ids = tf.placeholder(dtype=tf.int32, 
            shape=[None, self.config.max_n_cells], name="ids")
        self.features = tf.placeholder(dtype=tf.float32, 
            shape=[None, self.config.max_n_cells, self.config.n_features], name="features")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None])
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[])
        self.lr = tf.placeholder(dtype=tf.float32, shape=[])

        self.nodes = {
            "features": self.features, 
            "ids": self.ids}
        self.shapes = {
            "features": [None, self.config.max_n_cells, self.config.n_features], 
            "ids": [None, self.config.max_n_cells]}


class EmbeddingsInput(Model):
    def add_placeholder(self):
        """
        Defines self.x, self.y and self.dropout tf.placeholders
        """
        self.x = tf.placeholder(dtype=tf.float32, 
            shape=[None, self.config.max_n_cells, self.config.n_features], name="features")
        self.m = tf.placeholder(dtype=tf.bool, 
            shape=[None, self.config.max_n_cells], name="mask")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None])
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[])
        self.lr = tf.placeholder(dtype=tf.float32, shape=[])

        self.nodes = {"x": self.x}
        self.shapes = {"x": [None, self.config.max_n_cells, self.config.n_features]}


    def visualize(self, node_eval, logdir):
        
        tf.reset_default_graph()
        # summary_writer = tf.summary.FileWriter(logdir)
        # projector_config = projector.ProjectorConfig()

        embedding_var = tf.Variable(node_eval, "embeddings_viz")
        print embedding_var.get_shape()

        saver = tf.train.Saver()
        with tf.Session() as sess:       
            # embedding = projector_config.embeddings.add()
            # embedding.tensor_name = "embedding"
            # projector.visualize_embeddings(summary_writer, projector_config)
            print tf.global_variables()

            sess.run(tf.global_variables_initializer())
            emb_eval = sess.run([embedding_var])
            saver.save(sess, os.path.join(logdir, "model.ckpt"), 0)



   