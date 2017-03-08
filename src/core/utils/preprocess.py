import numpy as np


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


def pad_sequences(sequences, max_length, pad_tok):
    """
    Args:
        sequences: generator of sequences (list) of different length
        max_length: (int) max length of sequence
        pad_tok: same element as a sequence element
    Returns:
        np array of size [n_sequences, max_length, len(pad_took)]
    """
    result = []
    for seq in sequences:
        res_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        result +=  [res_]
    result = np.array(result)
    return result
