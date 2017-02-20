import numpy as np
import tensorflow as tf
import pickle
import time
from utils.tf_utils import xavier_weight_init, conv2d, \
                max_pool_2x2, weight_variable, bias_variable
from utils.general_utils import  Progbar, dump_results, \
                pickle_dump, pickle_load
from utils.preprocess_utils import minibatches, default_preprocess, \
                preprocess_data, split_data, baseline, one_hot, \
                load_and_preprocess_data, no_preprocess
from utils.features_utils import simple_features, get_simple_features
from utils.dataset import Dataset
import config


class Model(object):
    def __init__(self, config):
        self.config = config

    def add_placeholder(self):
        """
        Defines self.x and self.y, tf.placeholders
        """
        raise NotImplementedError

    def get_feed_dict(self, x, y=None):
        """
        Return feed dict
        """
        feed = {self.x: x}
        if y is not None:
            feed[self.y] = y
        return feed

    def add_prediction_op(self):
        """
        Defines self.pred
        """
        raise NotImplementedError

    def add_loss_op(self):
        """
        Defines self.loss
        """
        raise NotImplementedError

    def add_train_op(self):
        """
        Defines self.train_op
        """
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.train_op = optimizer.minimize(self.loss)


    def add_accuracy(self):
        """
        Defines self.accuracy, self.target, self.label
        """
        self.target = tf.argmax(self.y, axis=1)
        self.label = tf.argmax(self.pred, axis=1)
        self.correct_prediction = tf.equal(self.label, self.target)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def add_init(self):
        self.init = tf.global_variables_initializer()

    def build(self):
        print "Building model"

        self.add_placeholder()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_accuracy()
        self.add_init()

        print "- done."


    def run_epoch(self, sess, epoch, train_examples, dev_set, dev_baseline):
        print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
        prog = Progbar(target=1 + len(train_examples[0]) / config.batch_size)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, 
                                                self.config.batch_size)):
            _, train_loss = sess.run([self.train_op, self.loss], 
                                    feed_dict=self.get_feed_dict(train_x, train_y))
            prog.update(i + 1, [("train loss", train_loss)])

        print "Evaluating on dev set"
        acc, = sess.run([self.accuracy], feed_dict=self.get_feed_dict(dev_set[0], dev_set[1]))
        print "- dev acc: {:.2f} (baseline {:.2f})".format(acc * 100.0, 
                                                 dev_baseline * 100.0)

    def train(self, train_examples, dev_set, dev_baseline, test_set, test_baseline):
        with tf.Session() as sess:
            sess.run(self.init)
            print 80 * "="
            print "TRAINING"
            print 80 * "="

            for epoch in range(self.config.n_epochs):
               self.run_epoch(sess, epoch, train_examples, dev_set, dev_baseline)

            print 80 * "="
            print "TESTING"
            print 80 * "="
            print "Final evaluation on test set"
            acc, tar, lab = sess.run([self.accuracy, self.target, self.label], 
                            feed_dict=self.get_feed_dict(test_set[0], test_set[1]))
            dump_results(tar, lab, self.config.results_path)
            print "- test acc: {:.2f} (baseline {:.2f})".format(acc * 100.0, test_baseline * 100)

