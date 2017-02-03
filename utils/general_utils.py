import sys
import time
import numpy as np
import pickle

def my_print(string, level=2, verbose=0):
    """
    Prints string if level >= verbose
    """
    if level >= verbose:
        print(string)

import numpy as np

def get_my_print(verbose):
    """
    Returns lambda function to print with given verbose level
    """
    return lambda s, l=2: my_print(s, l, verbose)

def pickle_dump(obj, path):
    with open(path, "w") as f:
        pickle.dump(obj, f)

def pickle_load(path):
    with open(path) as f:
        return pickle.load(f)

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


def preprocess_data(data, preprocess, output_size):
    """
    Preprocess data with function preprocess
    Args:
        data: object on which we can iterate
        preprocess: (function) np array -> np array
        output_size: (int) for one hot encoding
    Returns:
        list of [x, y] where x, y are np arrays
    """
    x = np.array([d[0] for d in data])
    y = np.array([min(d[1], output_size-1) for d in data])
    x = preprocess(x)
    one_hot = np.zeros((y.size, output_size))
    one_hot[np.arange(y.size),y] = 1
    return [x, one_hot]

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

def baseline(data, target=1):
    """
    Return fraction of data example with label equal to target
    Args:
        data: [x, y] where x, y are np arrays
        traget: (int) the class target
    """

    return np.mean(np.argmax(data[1], axis=1) == 1)

def dump_results(target, label, path):
    """
    Writes results in a txt file
    Args:
        target: np array of the true labels [1, 2, 1, 1 ...]
        label: np array of the predicted labels
        path: path where to write the results
    """
    with open(path, "w") as f:
        f.write("True Pred\n")
        for t, l in zip(target, label):
            f.write("{}    {}\n".format(t, l))

class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)


