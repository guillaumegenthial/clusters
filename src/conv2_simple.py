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
from utils.features_utils import Extractor, simple_features, \
                cnn_simple_features
from utils.dataset import Dataset
import config


# data
print 80 * "="
print "INITIALIZING"
print 80 * "="

print "Loading data"
t0 = time.time()
extractor = Extractor(config.layer_extractors, config.modes)
n_layers = len(config.layer_extractors)
n_phi = config.layer_extractors.values()[0].n_phi
n_eta = config.layer_extractors.values()[0].n_eta
features = cnn_simple_features(extractor)

train_examples, dev_set, test_set = load_and_preprocess_data(config, 
                        features, no_preprocess, one_hot)
print "Train set shape: {}".format(train_examples[0].shape)
print "Dev   set shape: {}".format(dev_set[0].shape)
print "Test  set shape: {}".format(test_set[0].shape)
extractor.generate_report()
dev_baseline  = baseline(dev_set)
test_baseline = baseline(test_set)
t1 = time.time()
print "- done. (time elapsed {:.2f})".format(t1 - t0)


print "Building model"
x = tf.placeholder(tf.float32, shape=[None, n_phi, n_eta, n_layers*len(config.modes)])
y = tf.placeholder(tf.int32, shape=[None, config.output_size])
print n_phi, n_eta


W_conv1 = weight_variable([5, 5, n_layers*len(config.modes), 10])
b_conv1 = bias_variable([10])
h_conv1  = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_pool1_shape = [(n_phi + 1)/2, (n_eta + 1) /2]

W_conv2 = weight_variable([5, 5, 10, 5])
b_conv2 = bias_variable([5])
h_conv2  = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_shape = [(h_pool1_shape[0] + 1)/2, (h_pool1_shape[1] + 1)/2]
h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool2_shape[0]*h_pool2_shape[1]*5])

W_fc1 = weight_variable([h_pool2_shape[0]*h_pool2_shape[1] * 5, 10])
b_fc1 = bias_variable([10])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

W_fc2 = weight_variable([10, config.output_size])
b_fc2 = bias_variable([config.output_size])

pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

losses = tf.nn.softmax_cross_entropy_with_logits(pred, y)
loss = tf.reduce_mean(losses)

target = tf.argmax(y, axis=1)
label = tf.argmax(pred, axis=1)
correct_prediction = tf.equal(label, target)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_op = tf.train.AdamOptimizer(config.lr).minimize(loss)
init = tf.global_variables_initializer()

print "- done."


with tf.Session() as sess:
    sess.run(init)
    print 80 * "="
    print "TRAINING"
    print 80 * "="

    for epoch in range(config.n_epochs):
        print "Epoch {:} out of {:}".format(epoch + 1, config.n_epochs)
        prog = Progbar(target=1 + len(train_examples[0]) / config.batch_size)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, 
                                                config.batch_size)):
            feed = {x: train_x, y: train_y}
            _, train_loss = sess.run([train_op, loss], feed_dict=feed)
            prog.update(i + 1, [("train loss", train_loss)])

        print "Evaluating on dev set"
        feed = {x: dev_set[0], y: dev_set[1]}
        acc, = sess.run([accuracy], feed_dict=feed)
        print "- dev acc: {:.2f} (baseline {:.2f})".format(acc * 100.0, 
                                                 dev_baseline * 100.0)

    print 80 * "="
    print "TESTING"
    print 80 * "="
    print "Final evaluation on test set"
    feed = {x: test_set[0], y: test_set[1]}
    acc, tar, lab = sess.run([accuracy, target, label], feed_dict=feed)
    dump_results(tar, lab, config.results_path)
    print "- test acc: {:.2f} (baseline {:.2f})".format(acc * 100.0, test_baseline * 100)


