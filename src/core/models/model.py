import numpy as np
import tensorflow as tf
import time
import os
from shutil import copyfile
from datetime import datetime
import logging
from core.utils.tf_utils import xavier_weight_init, conv2d, \
                max_pool_2x2, weight_variable, bias_variable
from core.utils.general_utils import  Progbar, dump_results, export_matrices, \
                outputConfusionMatrix, check_dir
from core.utils.preprocess_utils import minibatches, baseline, \
                default_post_process
from core.utils.features_utils import Extractor

logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(message)s', level=logging.DEBUG)

class Model(object):
    def __init__(self, config, layers=None):
        self.config = config
        if self.config.output_path is None:
            self.config.output_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        check_dir(self.config.output_path)
        self.config.model_output = self.config.output_path + "model.weights/"
        self.config.eval_output = self.config.output_path + "results.txt"
        self.config.confmatrix_output = self.config.output_path + "confusion_matrix.png"
        self.config.config_output = self.config.output_path + "config.py"
        self.config.plot_output = self.config.output_path + "plots/"
        self.config.log_output = self.config.output_path + "log"

        if layers is None:
            self.layers = self.config.layers
        else:
            self.layers = layers

        handler = logging.FileHandler(self.config.log_output)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)


    def add_placeholder(self):
        """
        Defines self.x and self.y, tf.placeholders
        Defines self.x_shape of type [d1, d2, ...]
        """
        raise NotImplementedError

    def get_feed_dict(self, x, d, y=None):
        """
        Return feed dict
        """
        feed = {self.x: x,
                self.dropout: d}
        if y is not None:
            feed[self.y] = y
        return feed

    def add_prediction_op(self):
        """
        Defines self.pred
        """
        pred = self.x
        input_shape = self.x_shape
        for layer in self.layers:
            print "- at layer {}".format(layer.name)
            layer.set_param(dropout=self.dropout, input_shape=input_shape)
            pred = layer(pred)
            input_shape = layer.output_shape

        self.pred = pred


    def add_loss_op(self):
        """
        Defines self.loss
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y)
        self.loss = tf.reduce_mean(losses) + self.l2_loss() * self.config.reg


    def l2_loss(self):
        variables = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in variables
                    if 'bias' not in v.name ])

        return l2_loss

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
        if tf.__version__ > 0.12:
            self.label = tf.argmax(self.pred, axis=1)
        else:
            self.label = tf.argmax(self.pred, dimension=1)
            
        self.correct_prediction = tf.equal(tf.cast(self.label, tf.int32), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def add_init(self):
        self.init = tf.global_variables_initializer()

    def build(self):
        logger.info("Building model")

        self.add_placeholder()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_accuracy()
        self.add_init()

        logger.info("- done.")

    def save(self, saver, sess, path):
        """
        Save model to path
        """
        if not os.path.exists(path):
            os.makedirs(path)
        saver.save(sess, self.config.model_output)


    def run_epoch(self, sess, epoch, train_examples, dev_set, dev_baseline, post_process):
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
        logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
        prog = Progbar(target=(len(train_examples) + self.config.batch_size - 1) / self.config.batch_size)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, 
                                                self.config.batch_size)):

            train_x, train_y = post_process(train_x, train_y)
            fd = self.get_feed_dict(train_x, self.config.dropout, train_y)
            _, train_loss = sess.run([self.train_op, self.loss], feed_dict=fd)
            prog.update(i + 1, [("train loss", train_loss)])

        logger.info("Evaluating on dev set")
        dev_x, dev_y = zip(*dev_set)
        dev_x, dev_y = post_process(dev_x, dev_y)
        fd = self.get_feed_dict(dev_x, 1.0, dev_y)
        acc, = sess.run([self.accuracy], feed_dict=fd)
        logger.info("- dev acc: {:.2f} (baseline {:.2f})".format(acc * 100.0, 
                                                 dev_baseline * 100.0))
        return acc

    def train(self, train_examples, dev_set, post_process=default_post_process):
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
        self.export_config()
        with tf.Session() as sess:
            sess.run(self.init)
            logger.info(80 * "=")
            logger.info("TRAINING")
            logger.info(80 * "=")
            logger.info("- reg: {:.6f}, lr: {:.6f}".format(self.config.reg, self.config.lr))

            for epoch in range(self.config.n_epochs):
                acc = self.run_epoch(sess, epoch, train_examples, dev_set, dev_baseline, post_process)
                if acc > best_acc:
                    logger.info("- new best score! saving model in {}".format(self.config.model_output))
                    best_acc = acc
                    self.save(saver, sess, self.config.model_output)

    def evaluate(self, test_set, test_raw=None, post_process=default_post_process):
        """
        Reload weights and test on test set
        """
        logger.info(80 * "=")
        logger.info("TESTING")
        logger.info(80 * "=")
        logger.info("Final evaluation on test set")
        test_baseline = baseline(test_set)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init)
            saver.restore(sess, self.config.model_output)
            test_x, test_y = zip(*test_set)
            test_x, test_y = post_process(test_x, test_y)
            fd = self.get_feed_dict(test_x, 1.0, test_y)
            acc, lab = sess.run([self.accuracy, self.label], feed_dict=fd)
            logger.info("- test acc: {:.2f} (baseline {:.2f})".format(acc * 100.0, test_baseline * 100))
            outputConfusionMatrix(test_y, lab, self.config.output_size, self.config.confmatrix_output)
            if test_raw is not None:
                self.export_results(test_y, lab, test_raw)

        return acc, test_baseline

    def find_best_reg_value(self, reg_values, train_examples, dev_set, test_set, post_process=default_post_process):
        """
        Train model for different values of reg
        Args:
            reg_values: iterable
            ...
        Returns:
            list of tuples [(reg_value, score), ...]
        """
        result = []
        for reg in reg_values:
            self.config.reg = reg
            self.train(train_examples, dev_set)
            acc = self.evaluate(test_set, post_process=post_process)
            result += [(reg, acc)]

        self.get_reg_summary(result)
        return result

    def get_reg_summary(self, res):
        """
        Get reg summary
        Args:
            res: list of tuples [(reg_value, score), ...]
        """
        logger.info("="*80)
        logger.info("REG-SUMMARY")
        logger.info("="*80)
        for (reg, acc) in res:
            logger.info("Reg: {:.6f} Acc: {:.2f}".format(reg, acc*100))

        reg, acc = max(res, key=lambda (a, b): b)
        logger.info("- best reg value {:.6f} acc {:.2f}".format(reg, acc*100))


    def export_config(self):
        """
        Copies config file to outputpath
        """
        copyfile(self.config.__file__.split(".")[-2]+".py", self.config.config_output)

    def export_result(self, tar, lab, data_raw, extractor):
        """
        Export matrices of input
        """
        path = self.config.plot_output+ "true_{}_pred{}/".format(tar, lab)
        check_dir(path)
        matrices = extractor(data_raw["topo_cells"], data_raw["topo_eta"], 
                             data_raw["topo_phi"])
        export_matrices(matrices, path)

    def export_results(self, tar, lab, test_raw=None):
        """
        Export confusion matrix
        Export matrices for all pairs (tar, lab)
        """
        if test_raw is not None:
            extractor = Extractor(self.config.layer_extractors)
            tar_lab_seen = set()
            for (t, l, d_) in zip(tar, lab, test_raw):
                if (t, l) not in tar_lab_seen:
                    tar_lab_seen.add((t, l))
                    self.export_result(t, l, d_, extractor)
                    logger.info("- extracted layers for true label {}, pred {} in {}".format(
                                                t, l, self.config.plot_output))


