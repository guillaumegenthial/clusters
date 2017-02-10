import numpy as np
import tensorflow as tf
import pickle
import time
from utils.tf_utils import xavier_weight_init
from utils.general_utils import  Progbar, dump_results, \
                pickle_dump, pickle_load
from utils.preprocess_utils import minibatches, default_preprocess, \
                preprocess_data, split_data, baseline, one_hot, \
                load_and_preprocess_data
from utils.features_utils import simple_features
from utils.dataset import Dataset
import config

# model
x = tf.placeholder(tf.float32, shape=[None, config.input_size])
y = tf.placeholder(tf.int32, shape=[None, config.output_size])

W = tf.get_variable('W', (config.input_size, config.hidden_size), 
                        initializer=xavier_weight_init())

b = tf.get_variable('b', [config.hidden_size])
pred = tf.matmul(x, W) + b
# h = tf.nn.relu(tf.matmul(x, W) + b)
# h_drop = tf.nn.dropout(h, config.dropout)

# W2 = tf.get_variable('W2', (config.hidden_size, config.output_size),
#                         initializer=xavier_weight_init())
# b2 = tf.get_variable('b2', [config.output_size])
# pred = tf.matmul(h_drop, W2) + b2

losses = tf.nn.softmax_cross_entropy_with_logits(pred, y)
loss = tf.reduce_mean(losses)

target = tf.argmax(y, axis=1)
label = tf.argmax(pred, axis=1)
correct_prediction = tf.equal(label, target)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_op = tf.train.AdamOptimizer(config.lr).minimize(loss)
init = tf.global_variables_initializer()

# data
print 80 * "="
print "INITIALIZING"
print 80 * "="

t0 = time.time()
train_examples, dev_set, test_set = load_and_preprocess_data(config, 
                        simple_features, default_preprocess, one_hot)

dev_baseline  = baseline(dev_set)
test_baseline = baseline(test_set)
t1 = time.time()
print "- done. (time elapsed {:.2f})".format(t1 - t0)

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

