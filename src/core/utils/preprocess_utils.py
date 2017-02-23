import sys
import time
import numpy as np
import pickle
from general_utils import pickle_dump, pickle_load

def baseline(data, target=1):
    """
    Return fraction of data example with label equal to target
    Args:
        data: list of (x, y)
        traget: (int) the class target
    """
    y = zip(*data)[1]
    return np.mean(np.argmax(y, axis=1) == 1)

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
        x, y = np.asarray(x), np.asarray(y)
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
    if not config.load_from_export_data_path:
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
    
    if not config.load_from_export_data_path:
        data = load_data_raw_it(config)
        data = extract_data_it(data, featurizer)
        data = it_to_list(data)
        pickle_dump(data, data_path)

    else:
        data = pickle_load(data_path)

    return data


def it_to_list(data):
    """
    Args:
        data: generator that yields x, y
    Returns:
        list of (x, y)
    """
    print "Converting generator of (x, y) to list..."
    res = []
    for x, y in data:
        res.append((x, y))
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
    for d in data:
        x = featurizer(d)
        y = d["nparts"]
        yield x, y

def preprocess_data(data, preprocess_x, preprocess_y):
    """
    Preprocess data
    Args:
        data: list of tuples (x, y)
    Returns:
        list of tuples (x, y)
    """
    x, y = zip(*data)
    x = preprocess_x(x)
    y = preprocess_y(y)
    return zip(x, y)

def load_and_preprocess_data(config, featurizer=None, preprocess_x=None, preprocess_y=None, featurized=False):
    """
    Return train, dev and test set from data
    Args:
        config: config variables
        featurizer: function that takes as input the type 
            returned by the iterator on the data and computes
            features
        preprocess_x: function that takes as input the type
            returned by the featurizer
        featurized: (bool) if True, load the already featurized data
    Returns:
        train, dev and test set, list of (x, y)
    """
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="

    print "Loading data"
    t0 = time.time()

    if not featurized:

        data = load_data_raw(config)

        train_raw, dev_raw, test_raw = split_data(data, config.dev_size, config.test_size)

        train_ = extract_data(train_raw, featurizer)
        dev_ = extract_data(dev_raw, featurizer)
        test_ = extract_data(test_raw, featurizer)

    else:
        print "Warning: loading featurized data, test_raw = None"
        data = load_data_featurized(config, featurizer)
        train_, test_, dev_ = split_data(data, config.dev_size, config.test_size)
        test_raw = None

    if preprocess_x is not None and preprocess_y is not None:
        train_examples = preprocess_data(train_, preprocess_x, preprocess_y)
        dev_set        = preprocess_data(dev_, preprocess_x, preprocess_y)
        test_set       = preprocess_data(test_, preprocess_x, preprocess_y)
    else:
        print "Warning: no preprocessing applied to the data"

    print "    Train set shape: ({}, {})".format(
        len(train_examples), ", ".join([str(s) for s in train_examples[0][0].shape]))
    print "    Dev   set shape: ({}, {})".format(
        len(dev_set), ", ".join([str(s) for s in dev_set[0][0].shape]))
    print "    Test  set shape: ({}, {})".format(
        len(test_set), ", ".join([str(s) for s in test_set[0][0].shape]))

    t1 = time.time()
    print "- done. (time elapsed {:.2f})".format(t1 - t0)
    
    return train_examples, dev_set, test_set, test_raw

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