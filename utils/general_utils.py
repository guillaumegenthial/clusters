import sys
import time
import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt

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

def export_matrices(matrices, path="plots", vmin=-50, vmax=1000):
    """
    Saves an image of each matrix
    Args:
        matrices: dict of np arrays d[no of layer] = np array
        path: string to directory
        v: range of the plot colors
    """
    for i_, m_ in matrices.iteritems():
        plt.figure()
        m = copy.deepcopy(m_)
        m[m == 0] = np.nan
        plt.imshow(m, interpolation='nearest', cmap="bwr",  vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.grid(True)
        plt.savefig(path+"/layer_{}.png".format(i_))
        plt.close()
        del m


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

