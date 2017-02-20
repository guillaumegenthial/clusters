import sys
import time
import numpy as np
import pickle
from general_utils import pickle_dump, pickle_load

def baseline(data, target=1):
    """
    Return fraction of data example with label equal to target
    Args:
        data: [x, y] where x, y are np arrays
        traget: (int) the class target
    """
    return np.mean(np.argmax(data[1], axis=1) == 1)

def minibatches(data, minibatch_size, shuffle=True):
    """
    Returns an iterator
    Args:
        data: (list of np array) data[0] = X, data[1] = y typically
        minibatch_size: (int) 
        shuffle: (bool)
    Returns:
        iterator: yields a list of np array of size < minibatch_size
    """
    data_size = len(data[0])
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [d[minibatch_indices] for d in data]


def load_data(config):
    data_path = "{}_{}k.npy".format(config.export_data_path, 
                                        config.max_events)
    if not config.load_from_export_data_path:
        from dataset import Dataset
        data = Dataset(path=config.data_path, tree=config.tree_name, 
                       max_iter=config.max_events, verbose=config.data_verbosity,
                       max_eta=config.max_eta, min_energy=config.min_energy)

        data = data.get_data()
        pickle_dump(data, data_path)
    else:
        data = pickle_load(data_path)

    return data


def split_data(data, dev=0.1, test=0.2):
    """
    Splits randomly data into train, dev and test set
    Args:
        data: [x, y] where x, y are np arrays
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

    train_raw = [data[i] for i in train_indices]
    dev_raw = [data[i] for i in dev_indices]
    test_raw = [data[i] for i in test_indices]

    return train_raw, dev_raw, test_raw


def extract_data(data, feature_extractor=lambda x: x):
    """
    Extract data with feature_extractor
    Args:
        data: object on which we can iterate 
    Returns:
        [x, y] where x, y are np arrays
    """
    x = np.array([feature_extractor(d) for d in data])
    y = np.array([d["nparts"] for d in data])
    return [x, y]

def preprocess_data(data, preprocess_x, preprocess_y):
    """
    Preprocess data
    Args:
        data: [x, y] where x and y are np arrays
    Returns:
        [x, y]
    """
    x = preprocess_x(data[0])
    y = preprocess_y(data[1])
    return [x, y]


def load_and_preprocess_data(config, extractor, preprocess_x, preprocess_y):
    """
    Return train, dev and test set from data
    Args:
        config: config variables
        extractor: function that takes as input the type 
            returned by the iterator on the data and computes
            features
        preprocess: function that takes as input the type
            returned by the extractor
    Returns:
        train, dev and test set
    """
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="

    print "Loading data"
    t0 = time.time()

    data = load_data(config)
    train_raw, dev_raw, test_raw = split_data(data, config.dev_size, config.test_size)

    train_examples = extract_data(train_raw, extractor)
    dev_set = extract_data(dev_raw, extractor)
    test_set = extract_data(test_raw, extractor)

    train_examples = preprocess_data(train_examples, preprocess_x, preprocess_y)
    dev_set = preprocess_data(dev_set, preprocess_x, preprocess_y)
    test_set = preprocess_data(test_set, preprocess_x, preprocess_y)

    print "    Train set shape: {}".format(train_examples[0].shape)
    print "    Dev   set shape: {}".format(dev_set[0].shape)
    print "    Test  set shape: {}".format(test_set[0].shape)

    t1 = time.time()
    print "- done. (time elapsed {:.2f})".format(t1 - t0)
    
    return train_examples, dev_set, test_set, test_raw


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
    X /= sigma(X)
    return X