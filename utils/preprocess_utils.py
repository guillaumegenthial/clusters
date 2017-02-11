import sys
import time
import numpy as np
import pickle
from dataset import Dataset
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


def preprocess_data(data, preprocess_x, preprocess_y, 
            output_size, feature_extractor=lambda x: x):
    """
    Preprocess data with function preprocess
    Args:
        data: object on which we can iterate and yields X, y
        preprocess_x: (function) np array -> np array
        preprocess_y: (function) np array -> np 
        output_size: (int) for one hot encoding
        feature_extractor: (function) data -> np array
    Returns:
        list of [x, y] where x, y are np arrays
    """
    x = np.array([feature_extractor(d) for d in data])
    y = np.array([min(d["nparts"], output_size-1) for d in data])
    x = preprocess_x(x)
    y = preprocess_y(y, output_size)
    return [x, y]

def one_hot(y, output_size):
    """
    Takes a label y an return a one-hot vector
    """
    one_hot = np.zeros((y.size, output_size))
    one_hot[np.arange(y.size),y] = 1
    return one_hot

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
    data_path = "{}_{}k.npy".format(config.export_data_path, 
                                        config.max_events)
    if not config.load_from_export_data_path:
        data = Dataset(config.data_path, config.tree_name, 
                       config.max_events, config.data_verbosity)
        data = data.get_data()
        pickle_dump(data, data_path)
    else:
        data = pickle_load(data_path)

    data = preprocess_data(data, preprocess_x, preprocess_y, 
                           config.output_size, extractor)
    train_examples, dev_set, test_set = split_data(data, config.dev_size,
                                                   config.test_size)

    return train_examples, dev_set, test_set


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
    data_size = len(data[0])
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    train_indices = indices[0:int(data_size*(1-dev-test))]
    dev_indices = indices[int(data_size*(1-dev-test)):int(data_size*(1-test))]
    test_indices = indices[int(data_size*(1-test)):]

    train_examples = [data[0][train_indices], data[1][train_indices]]
    dev_set = [data[0][dev_indices], data[1][dev_indices]]
    test_set = [data[0][test_indices], data[1][test_indices]]

    return train_examples, dev_set, test_set

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

def mean_substraction(X):
    """
    Substracts mean to a sample X
    Args:
        X: np array of shape (nsamples, nfeatures)
    Returns:
        X - mean(X)
    """

    X -= np.mean(X, axis=0, keepdims=True)
    return X

def normalization(X):
    """
    Divides X by its standard deviation
    Args:
        X: np array of shape (nsamples, nfeatures)
    Returns:
        X / np.sqrt(np.var(X))
    """
    X /= np.sqrt(np.var(X, axis=0, keepdims=True))
    return X