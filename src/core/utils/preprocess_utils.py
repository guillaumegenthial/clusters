import sys
import time
import os
import copy
import numpy as np
import pickle
from general_utils import pickle_dump, pickle_load, Progbar


def minibatches(data, minibatch_size, shuffle=True):
    """
    Returns an iterator
    Args:
        data: list of x, y
        minibatch_size: (int) 
        shuffle: (bool)
    Returns:
        iterator: yields a list of np array of size < minibatch_size
    """
    data_size = len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        x, y = zip(*([data[i] for i in minibatch_indices]))
        yield x, y 


def load_data_raw(config):
    """
    Return data
    Args:
        config: modul config
    Retuns:
        list of raw data for each example
    """
    data_path = "{}_{}k.npy".format(config.export_data_path, 
                                        config.max_events)
    if not os.path.exists(data_path):
        data = load_data_raw_it(config)
        data = data.get_data()
        pickle_dump(data, data_path)
    else:
        data = pickle_load(data_path)

    return data

def load_data_raw_it(config):
    """
    It version of load_data_raw that yields raw data for each example
    """
    from dataset import Dataset
    data = Dataset(path=config.data_path, tree=config.tree_name, 
                   max_iter=config.max_events, verbose=config.data_verbosity,
                   max_eta=config.max_eta, min_energy=config.min_energy)
    return data

def load_data_featurized(config, featurizer=None):

    data_path = "{}_featurized_{}k.npy".format(config.export_data_path, 
                                        config.max_events)
    
    if not os.path.exists(data_path):
        data = load_data_raw_it(config)
        data = extract_data_it(data, featurizer)
        data = it_to_list(data, config)
        pickle_dump(data, data_path)

    else:
        data = pickle_load(data_path)

    return data


def it_to_list(data, config):
    """
    Args:
        data: generator that yields x, y
    Returns:
        list of (x, y)
    """
    print "Converting generator of (x, y) to list..."
    res = []
    prog = Progbar(target=config.max_events)
    count_ = 0
    for (x, y), i in data:
        count_ += 1
        res.append((x, y))
        if count_ % 20 == 0 and i < config.max_events - 1:
            prog.update(i + 1, strict=[("Sample no ", count_)])
            
    prog.update(i+1, strict=[("Sample no ", count_)])
    print "- done."
    return res

def split_data(data, dev=0.1, test=0.2):
    """
    Splits randomly data into train, dev and test set
    Args:
        data: iterable over each training example that yield (x, y)
        dev: (float) fraction of dev set 
        test: (float) fraction of test set
    Returns:
        (train_examples, dev_set, test_set) where each element is the same
                as data, a list of 2 np arrays [x, y]
    """
    data_size = len(data)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    train_indices = indices[0:int(data_size*(1-dev-test))]
    dev_indices = indices[int(data_size*(1-dev-test)):int(data_size*(1-test))]
    test_indices = indices[int(data_size*(1-test)):]

    train_ = [data[i] for i in train_indices]
    dev_ = [data[i] for i in dev_indices]
    test_ = [data[i] for i in test_indices]

    return train_, dev_, test_


def extract_data(data, extractor=lambda x: x):
    """
    Extract data with feature_extractor
    Args:
        data: object on which we can iterate 
    Returns:
        list of (x, y)
    """
    return [(extractor(d), d["nparts"]) for d in data]

def extract_data_it(data, featurizer=lambda x: x):
    """
    Iterate over data and return iterator yielding x and y
    Args:
        data: iterable that yields raw data
        feature extractor: takes raw data, return np array
    Returns:
        iterator that yields (x, y)
    """
    for d, i in data:
        x = featurizer(d)
        y = d["nparts"]
        yield (x, y), i

def make_preprocess(preprocess_x, preprocess_y):
    """
    Preprocess data
    Args:
        data: list of tuples (x, y)
    Returns:
        list of tuples (x, y)
    """
    def f(data):
        x, y = zip(*data)
        x = preprocess_x(x)
        y = preprocess_y(y)
        return zip(x, y)
        
    return f

def load_and_preprocess_data(config, featurizer, preprocess=None):
    """
    Return train, dev and test set from data
    Args:
        config: config variables
        featurizer: function that takes as input the type 
            returned by the iterator on the data and computes
            features
        preprocess: function that takes as input the type
            returned by the featurizer
    Returns:
        train, dev and test set, list of (x, y)
    """
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="

    print "Loading data"
    t0 = time.time()

    if not config.featurized:
        data = load_data_raw(config)

        train_raw, dev_raw, test_raw = split_data(data, config.dev_size, config.test_size)

        train_ = extract_data(train_raw, featurizer)
        dev_ = extract_data(dev_raw, featurizer)
        test_ = extract_data(test_raw, featurizer)

    else:
        print "Warning: loading featurized data, test_raw = test"
        data = load_data_featurized(config, featurizer)
        train_, test_, dev_ = split_data(data, config.dev_size, config.test_size)
        test_raw = copy.deepcopy(test_)

    if preprocess is not None:
        train_ = preprocess(train_)
        dev_   = preprocess(dev_)
        test_  = preprocess(test_)
    else:
        print "Warning: no preprocessing applied to the data"

    try:
        print "    Train set shape: ({}, {})".format(
            len(train_), ", ".join([str(s) for s in train_[0][0].shape]))
        print "    Dev   set shape: ({}, {})".format(
            len(dev_), ", ".join([str(s) for s in dev_[0][0].shape]))
        print "    Test  set shape: ({}, {})".format(
            len(test_), ", ".join([str(s) for s in test_[0][0].shape]))
    except:
        pass

    t1 = time.time()
    print "- done. (time elapsed {:.2f})".format(t1 - t0)
    
    return train_, dev_, test_, test_raw

def export_data(data, modes, path):
    """
    writes features to file
    Args:
        data: list of x, y
    """
    with open(path, "w") as f:
        f.write(", ".join(modes) + ", " + "nparts")
        f.write("\n")
        for x, y in data:
            feat = ", ".join(map(str, x))
            feat += ", " + str(y) + "\n"
            f.write(feat)

def one_hot(output_size):
    """
    Takes a label y an return a one-hot vector
    """
    def f(y):
        y = np.asarray([min(y_, output_size-1) for y_ in y])
        one_hot = np.zeros((y.size, output_size))
        one_hot[np.arange(y.size), y] = 1
        return one_hot
    return f

def max_y(output_size):
    def f(y):
        y = np.asarray([min(y_, output_size-1) for y_ in y])
        return y
    return f

def default_preprocess(X):
    """
    Preprocess X by mean substracting and normalization
    Args:
        X: (np array) of shape (nsamples, nfeatures)
    Returns:
        X: (np array) (X - m) / sigma
    """
    X = mean_substraction(X)
    X = normalization(X)
    return X

def default_post_process(X, Y):
    return np.asarray(X), np.asarray(Y)

def no_preprocess(X):
    return X

def scale_preprocess(scale):
    return lambda X: X/float(scale)

def mean(X):
    return np.mean(X, axis=0, keepdims=True)

def sigma(X):
    return np.sqrt(np.var(X, axis=0, keepdims=True))

def mean_substraction(X):
    """
    Substracts mean to a sample X
    Args:
        X: np array of shape (nsamples, nfeatures)
    Returns:
        X - mean(X)
    """

    X -= mean(X)
    return X

def normalization(X):
    """
    Divides X by its standard deviation
    Args:
        X: np array of shape (nsamples, nfeatures)
    Returns:
        X / np.sqrt(np.var(X))
    """
    eps = 10^(-6)
    X /= (sigma(X) + eps)
    return X