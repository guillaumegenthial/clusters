import numpy as np
import tensorflow as tf
import time
import os
import logging
from shutil import copyfile
from datetime import datetime
from core.utils.tf import xavier_weight_init, conv2d, \
                max_pool_2x2, weight_variable, bias_variable
from core.utils.general import  Progbar, check_dir, get_all_dirs
from core.utils.data import minibatches, get_xy
from core.utils.evaluate import raw_export_result, baseline, \
    outputConfusionMatrix, outputF1Score, dump_results, f1score, \
    outputPerfProp

logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(message)s', level=logging.DEBUG)

class Model(object):
    def __init__(self, config, layers=None):
        self.config = config
        if self.config.output_path is None:
            if self.config.restore:
                self.config.output_path = "results/{}/".format(max(get_all_dirs("results/")))
            else:
                self.config.output_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())

        check_dir(self.config.output_path)
        self.config.model_output = self.config.output_path + "model.weights/"
        self.config.eval_output = self.config.output_path + "results.txt"
        self.config.confmatrix_output = self.config.output_path + "confusion_matrix.png"
        self.config.perfleadprop_output = self.config.output_path + "perf_leadprop.png"
        self.config.config_output = self.config.output_path
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
        Defines self.shapes dict("x": shape of x, ...)
        Defines self.nodes dict("x": self.x, ...)
        """
        raise NotImplementedError

    def get_feed_dict(self, x, d, y=None, m=None):
        """
        Return feed dict
        Args:
            x: batch of x
            y: batch of y
            d: dropout keep prob
            m: batch of mask
        """
        feed = {self.x: x,
                self.dropout: d}
        if y is not None:
            feed[self.y] = y
        if m is not None and hasattr(self, "m"):
            feed[self.m] = m
        return feed

    def add_prediction_op(self):
        """
        Defines self.pred
        """
        # get inputs
        input_nodes = self.nodes.keys()
        # default
        pred = self.nodes[input_nodes[0]]
        input_shape = self.shapes[input_nodes[0]]
        # mask
        if hasattr(self, "m"):
            mask = self.m
        else:
            mask = None
        # go through the layers
        for layer in self.layers:
            # default
            if layer.input_names == []:
                print "- at layer {}, input shape {}".format(layer.name, input_shape)
                layer.set_param(dropout=self.dropout, input_shape=input_shape)
                pred = layer(pred, mask)
            else:
                input_shape = [self.shapes[n] for n in layer.input_names]
                print "- at layer {}, input shape {}".format(layer.name, input_shape)
                inputs = [self.nodes[input_name] for input_name in layer.input_names]
                if len(inputs) == 1:
                    inputs = inputs[0]
                    input_shape = input_shape[0]
                layer.set_param(dropout=self.dropout, input_shape=input_shape)
                pred = layer(inputs, mask)

            self.nodes[layer.name] = pred
            self.shapes[layer.name] = layer.output_shape
            print "- at layer {}, output shape {}".format(layer.name, layer.output_shape)
            # default
            input_shape = layer.output_shape

        self.pred = pred


    def add_loss_op(self):
        """
        Defines self.loss
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y)
        self.loss = tf.reduce_mean(losses) 
        # + self.l2_loss() * self.config.reg
        tf.summary.scalar("loss", self.loss)


    def l2_loss(self):
        with tf.variable_scope("l2_loss"):
            variables = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in variables
                        if 'bias' not in v.name ])

        return l2_loss

    def add_train_op(self):
        """
        Defines self.train_op
        """
        with tf.variable_scope("train_step"):
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

    def add_summary(self, sess): 
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, sess.graph)


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

    def run_baseline(self, test_set, processing=None):
        test_x, test_y, test_p = get_xy(test_set)
        if processing is not None:
            test_x, test_y, mask = processing(test_x, test_y)

        base_f1 = f1score(test_y, self.config.baseclass*np.ones(len(test_y)), 
            labels=range(self.config.output_size), average=self.config.f1_mode)
        base_acc = baseline(test_y, target=self.config.baseclass)
        return base_acc, base_f1

    def run_evaluate(self, sess, test_set, base_acc, base_f1, processing=None):
        """
        Computes evaluation over a test set an log it
        """
        ys, labs, accs, lead_props = [], [], [], []
        for (x, y, p) in minibatches(test_set, self.config.batch_size):
            if processing is not None:
                x, y, m = processing(x, y)

            fd = self.get_feed_dict(x, 1.0, y, m)
            acc, lab = sess.run([self.accuracy, self.label], feed_dict=fd)

            ys   += [y]
            labs += [lab]
            accs  += [acc]
            lead_props += [max(p_ if len(p_) != 0 else [0]) for p_ in p]

        acc = np.mean(accs)
        labs = np.concatenate(labs, axis=0)
        ys = np.concatenate(ys, axis=0)

        f1 = f1score(ys, labs, labels=range(self.config.output_size), average=self.config.f1_mode)
        logger.info("- dev acc: {:04.2f} (baseline {:04.2f}) f1: {:04.2f} (baseline {:04.2f})".format(
            acc * 100.0, base_acc * 100.0, f1 * 100.0, base_f1*100.0))

        return acc, f1, ys, labs, lead_props

    def run_epoch(self, sess, epoch, train_examples, dev_set, dev_baseline, dev_baseline_f1, processing=None):
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
        n_batches = (len(train_examples) + self.config.batch_size - 1) / self.config.batch_size
        logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
        prog = Progbar(target=n_batches)
        for i, (train_x, train_y, train_p) in enumerate(minibatches(train_examples, 
                                                self.config.batch_size)):

            if processing is not None:
                train_x, train_y, mask = processing(train_x, train_y)

            fd = self.get_feed_dict(train_x, self.config.dropout, train_y, mask)
            
            _, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)
            prog.update(i + 1, [("train loss", train_loss)])
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*n_batches + i)

            if time.time() - self.t_begin > self.config.max_training_time:
                logger.info("\nMax time elapsed. Early Stopping")
                break

        logger.info("Evaluating on dev set")
        
        acc, dev_f1, _, _, _ = self.run_evaluate(sess, dev_set, dev_baseline, dev_baseline_f1, processing)

        return acc, dev_f1

    def train(self, train_examples, dev_set, processing=None):
        """
        Train model and saves param
        Args:
            train_examples: data generator
            dev_set: data generator
            processing: function of batch tuples (X, Y)
        Returns:
            perform training operation on train_examples data with config
            computes performance at each epoch and saves model if best
        """
        best_score = 0
        nb_ep_no_imprvmt = 0
        self.t_begin = time.time()
        
        dev_base_acc, dev_base_f1 = self.run_baseline(dev_set, processing)

        saver = tf.train.Saver()
        self.export_config()
        with tf.Session() as sess:
            logger.info(80 * "=")
            logger.info("TRAINING")
            logger.info(80 * "=")
            logger.info("- reg: {:.6f}, lr: {:.6f}".format(self.config.reg, self.config.lr))

            self.add_summary(sess)
            sess.run(self.init)
            if self.config.restore:
                logger.info("- Restoring model from {}".format(self.config.model_output))
                saver.restore(sess, self.config.model_output)

            for epoch in range(self.config.n_epochs):
                acc, dev_f1 = self.run_epoch(sess, epoch, train_examples, dev_set, 
                    dev_base_acc, dev_base_f1, processing)

                score = dev_f1 if self.config.selection == "f1" else acc
                if score >= best_score:
                    nb_ep_no_imprvmt = 0
                    logger.info("- new best {}! saving model in {}".format(
                        self.config.selection, self.config.model_output))
                    best_score = score
                    self.save(saver, sess, self.config.model_output)
                else:
                    nb_ep_no_imprvmt += 1
                    if (self.config.early_stopping and nb_ep_no_imprvmt >= self.config.nb_ep_no_imprvmt):
                        logger.info("- {} epochs without improvement. Early Stopping".format(nb_ep_no_imprvmt))
                        break
                if time.time() - self.t_begin > self.config.max_training_time:
                    logger.info("\nMax time elapsed. Early Stopping")
                    break


                
    def evaluate(self, test_set, processing=None, test_raw=None, export_result=None):
        """
        Reload weights and test on test set
        Args:
            test_set: data generator
            processing: function to apply to batch tuples (X, Y)
            test_raw: data generator of raw data for export
            export_result: f(config, logger, ys, labs, test_raw)
        Return:
            confusion matrix
            f1 score of the baseline model
            f1 score of the model
            export plots of examples if test_raw and export result are defined
            acc, test_base_acc: accuracy of the model, acc of the baseline
        """
        logger.info(80 * "=")
        logger.info("TESTING")
        logger.info(80 * "=")
        logger.info("Final evaluation on test set")
        
        test_base_acc, test_base_f1 = self.run_baseline(test_set, processing)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init)
            logger.info("Restoring model from {}".format(self.config.model_output))
            saver.restore(sess, self.config.model_output)
            
            acc, test_f1, ys, labs, lead_props = self.run_evaluate(sess, test_set, 
                                test_base_acc, test_base_f1, processing)

            outputConfusionMatrix(ys, labs, self.config.part_min, 
                    self.config.output_size, self.config.confmatrix_output)
            outputF1Score(self.config, logger, ys, self.config.baseclass*np.ones(len(ys)), "Baseline")
            logger.info("\n")
            outputF1Score(self.config, logger, ys, labs, "Model")
            outputPerfProp(ys, labs, lead_props, self.config.perfleadprop_output, bins=5, 
                av=self.config.f1_mode, output_size=self.config.output_size)

            for eval_perf_class in range(0, self.config.output_size):
                filename = self.config.output_path + "perf_leadprop_{}.png".format(eval_perf_class)
                outputPerfProp(ys, labs, lead_props, filename,
                    bins=5, av=self.config.f1_mode, output_size=self.config.output_size, 
                    eval_perf_class=eval_perf_class, part_min=self.config.part_min)

            if test_raw is not None and export_result is not None:
                export_result(self.config, logger, ys, labs, test_raw)

        return acc, test_base_acc

    def find_best_reg_value(self, reg_values, train_examples, dev_set, test_set, processing=None):
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
            acc = self.evaluate(test_set, processing=processing)
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
        for file in self.config.config_files:
            copyfile(file, self.config.config_output + file.split("/")[-1])

    def eval_node(self, x, y, node_name, processing=None):
        """
        Evaluate model at node_name on data example x, y (single)
        Args:
            x: cells
            y: the label
            node_name: name of node to eval
            processing: function that take x, y batches and return x, y, mask
        Returns:
            evaluation of the nodes on the example x, y (only valid, extract mask)
        """
        # batch the x, y
        x, y = [x], [y]
       
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.config.model_output)
            if processing is not None:
                x, y, m = processing(x, y)

            fd = self.get_feed_dict(x, 1.0, y, m)
            lab, node_eval = sess.run([self.label, self.nodes[node_name].name], feed_dict=fd)

        return node_eval[0][:int(np.sum(m))], y, lab


    