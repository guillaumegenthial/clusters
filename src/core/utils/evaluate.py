import copy
import itertools
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, \
    precision_score, recall_score
from core.features.layers import Extractor
from general import check_dir

# deactivate undefined metric warning from sklearn
import warnings
from sklearn.metrics.classification import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def f1score(*args, **kwargs):
    return f1_score(*args, **kwargs)


def baseline(y, target=1):
    """
    Return fraction of data example with label equal to target
    Args:
        data: list of y
        traget: (int) the class target
    """
    return np.mean(np.asarray(y) == 1)


def export_matrices(matrices, path="plots/", suffix="", vmin=-50, vmax=1000):
    """
    Saves an image of each matrix
    Args:
        matrices: dict of np arrays d[no of layer] = np array
        path: string to directory
        v: range of the plot colors
    """
    for i_, modes_ in matrices.iteritems():
        for mode_, m_ in modes_.iteritems():
            if type(m_) != tuple and np.sum(m_) != 0:
                plt.figure()
                m = copy.deepcopy(m_)
                m[m == 0] = np.nan
                # plt.imshow(m, interpolation='nearest', cmap="bwr",  vmin=vmin, vmax=vmax)
                plt.imshow(m, interpolation='nearest', cmap="bwr")
                plt.colorbar()
                plt.grid(True)
                plt.savefig(path+"layer_{}_{}{}.png".format(i_, mode_, suffix))
                plt.close()
                del m


def raw_export_result(config, logger):

    def export_result(tar, lab, data_raw, extractor):
        """
        Export matrices of input
        """
        path = config.plot_output+ "true_{}_pred{}/".format(tar, lab)
        check_dir(path)
        matrices = extractor(data_raw["topo_cells"], data_raw["topo_eta"], 
                             data_raw["topo_phi"])
        export_matrices(matrices, path)

    def export_results(tar, lab, test_raw=None):
            """
            Export confusion matrix
            Export matrices for all pairs (tar, lab)
            """
            if test_raw is not None:
                extractor = Extractor(config.layer_extractors)
                tar_lab_seen = set()
                for (t, l, d_) in zip(tar, lab, test_raw):
                    if (t, l) not in tar_lab_seen:
                        logger.info("- extracting layers for true label {}, pred {} in {}".format(
                                                    t, l, config.plot_output))
                        tar_lab_seen.add((t, l))
                        export_result(t, l, d_, extractor)
                        

    return export_results

def featurized_export_result(config, logger):

    def export_results(tar, lab, test_raws=None):
            """
            Export confusion matrix
            Export matrices for all pairs (tar, lab)
            """
            if test_raws is not None:
                tar_lab_seen = set()
                test_raw, _ = zip(*test_raws[0])
                test_set, _ = zip(*test_raws[1])
                for (t, l, d_, d) in zip(tar, lab, test_raw, test_set):
                    if (t, l) not in tar_lab_seen:
                        logger.info("- extracting layers for true label {}, pred {} in {}".format(
                                                    t, l, config.plot_output))
                        tar_lab_seen.add((t, l))
                        path = config.plot_output + "true_{}_pred{}/".format(t, l)
                        check_dir(path)
                        export_matrices(d_, path, "_raw")
                        export_matrices(d, path, "_preprocessed")
                        

    return export_results


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


def outputConfusionMatrix(tar, lab, output_size, filename):
    """ Generate a confusion matrix """
    cm = confusion_matrix(tar, lab, labels=range(output_size))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    classes = range(output_size)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.close()

def outputF1Score(config, logger, tar, lab, name, labels=None):
    if labels is None:
        labels = range(config.output_size)  

    averages = ["micro", "macro", "weighted"]

    logger.info("\t".join([name + " "*(len(max(averages, key=len)) - len(name)), 
                        "Precision", "Recall", "F1"]))
    for av in averages:

        f1 = f1_score(tar, lab, labels=labels, average=av)
        rc = recall_score(tar, lab, labels=labels, average=av)
        pr = precision_score(tar, lab, labels=labels, average=av)

        av_toprint = av + " "*(len(max(averages, key=len)) - len(av))
        logger.info("\t".join([av_toprint, "{:.4}".format(pr*100.0)+" "*(len("Precision")-4), 
            "{:.4}".format(rc*100.0)+" "*(len("Recall")-4), "{:.4}".format(f1*100.0)]))


    f1_global = f1_score(tar, lab, labels=labels, average=None)
    pr_global = precision_score(tar, lab, labels=labels, average=None)
    rc_global = recall_score(tar, lab, labels=labels, average=None)

    for (pr, rc, f1, lab) in zip(pr_global, rc_global, f1_global, labels):
        lab_toprint = "lab {}".format(lab) + " "*(len(max(averages, key=len)) - len("lab {}".format(lab)))
        logger.info("\t".join([lab_toprint, "{:.4}".format(pr*100.0)+" "*(len("Precision")-4), 
            "{:.4}".format(rc*100.0)+" "*(len("Recall")-4), "{:.4}".format(f1*100.0)]))


   

def simplePlot(x, ys, filename, xlabel="", ylabel=""):
    plt.figure()
    ys = zip(*ys)
    for y in ys:
        plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()


