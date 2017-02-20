import numpy as np
import tensorflow as tf
import time
import os
from shutil import copyfile
from datetime import datetime
from utils.tf_utils import xavier_weight_init, conv2d, \
                max_pool_2x2, weight_variable, bias_variable
from utils.general_utils import  Progbar, dump_results, export_matrices, \
                outputConfusionMatrix
from utils.preprocess_utils import minibatches, baseline
from utils.features_utils import Extractor


class Model(object):
    def __init__(self, config):
        self.config = config
        if self.config.output_path is None:
            self.config.output_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.config.model_output = self.config.output_path + "model.weights/"
        self.config.eval_output = self.config.output_path + "results.txt"
        self.config.conf_matrix = self.config.output_path + "confusion_matrix.png"
        self.config.plot_output = self.config.output_path + "plots/"
        self.config.log_output = self.config.output_path + "log"

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

    def save(self, saver, sess, path):
        """
        Save model to path
        """
        if not os.path.exists(path):
            os.makedirs(path)
        saver.save(sess, self.config.model_output)


    def run_epoch(self, sess, epoch, train_examples, dev_set, dev_baseline):
        """
        Run one epoch of training
        Args:
            sess: tf.session
            epoch: (int) index of epoch
            train_examples: (nsamples, nfeatures) np array
            dev_set: ...
            dev_baseline: float, acc from baseline
        Returns:
            acc: accuracy on dev set
        """
        print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
        prog = Progbar(target=1 + len(train_examples[0]) / self.config.batch_size)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, 
                                                self.config.batch_size)):
            _, train_loss = sess.run([self.train_op, self.loss], 
                                    feed_dict=self.get_feed_dict(train_x, train_y))
            prog.update(i + 1, [("train loss", train_loss)])

        print "Evaluating on dev set"
        acc, = sess.run([self.accuracy], feed_dict=self.get_feed_dict(dev_set[0], dev_set[1]))
        print "- dev acc: {:.2f} (baseline {:.2f})".format(acc * 100.0, 
                                                 dev_baseline * 100.0)
        return acc

    def train(self, train_examples, dev_set):
        """
        Train model and saves param
        Args:
            train_examples: (nsamples, nfeatures) np array
            dev_set: ...
            dev_baseline: float, acc from baseline
        """
        best_acc = 0
        dev_baseline = baseline(dev_set)
        saver = tf.train.Saver()

        copyfile(self.config.config_file, self.config.output_path+"config.py")
        with tf.Session() as sess:
            sess.run(self.init)
            print 80 * "="
            print "TRAINING"
            print 80 * "="

            for epoch in range(self.config.n_epochs):
               acc = self.run_epoch(sess, epoch, train_examples, dev_set, dev_baseline)
               if acc > best_acc:
                print "- new best score! saving model in ", self.config.model_output
                self.save(saver, sess, self.config.model_output)

    def evaluate(self, test_set, test_raw=None):
        """
        Reload weights and test on test set
        """
        print 80 * "="
        print "TESTING"
        print 80 * "="
        print "Final evaluation on test set"
        test_baseline = baseline(test_set)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init)
            saver.restore(sess, self.config.model_output)
            acc, tar, lab = sess.run([self.accuracy, self.target, self.label], 
                            feed_dict=self.get_feed_dict(test_set[0], test_set[1]))
            print "- test acc: {:.2f} (baseline {:.2f})".format(acc * 100.0, test_baseline * 100)
            self.export_results(tar, lab, test_raw)


    def export_result(self, tar, lab, data_raw, extractor):
        path = self.config.plot_output+ "tar_{}_label_{}/".format(tar, lab)
        if not os.path.exists(path):
            os.makedirs(path)
        matrices = extractor(data_raw["topo_cells"], data_raw["topo_eta"], 
                             data_raw["topo_phi"])
        export_matrices(matrices, path)

    def export_results(self, tar, lab, test_raw=None):
        """
        Export result
        """
        # dump_results(tar, lab, self.config.eval_output)
        outputConfusionMatrix(tar, lab, self.config.output_size, self.config.conf_matrix)
        if test_raw is not None:
            extractor = Extractor(self.config.layer_extractors)
            tar_lab_seen = set()
            for (t, l, d_) in zip(tar, lab, test_raw):
                if (t, l) not in tar_lab_seen:
                    tar_lab_seen.add((t, l))
                    self.export_result(t, l, d_, extractor)
                    print "- extracted layers for target {} label {} in {}".format(
                                                t, l, self.config.plot_output)


