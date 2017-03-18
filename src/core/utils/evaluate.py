import copy
import itertools
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, \
    precision_score, recall_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from core.features.layers import Extractor
from general import check_dir
from core.features.layers import get_custom_extractor


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
    return np.mean(np.asarray(y) == target)


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
                if mode_ != "cluster":
                    plt.imshow(m, interpolation='nearest', cmap="bwr")
                else:
                    plt.imshow(m, interpolation='nearest', cmap="Accent",  vmin=1, vmax=vmax)
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

def featurized_export_result(config, logger, tar, lab, test_raw=None):
    """
    Export matrices for all pairs (tar, lab)
    Args:
        config: config file
        logger: logging object
        tar: true labels
        lab: predicted labels
        test_raw: featurized inputs
    """
    if test_raw is not None:
        tar_lab_seen = set()
        for (t, l, d) in zip(tar, lab, test_raw):
            data_tuple, _ = d
            x, y = data_tuple[0], data_tuple[1]
            if (t, l) not in tar_lab_seen:
                logger.info("- extracting layers for true label {}, pred {} in {}".format(
                                            t+config.part_min, l+config.part_min, config.plot_output))
                tar_lab_seen.add((t, l))
                path = config.plot_output + "true_{}_pred{}/".format(t+config.part_min, l+config.part_min)
                check_dir(path)
                export_matrices(x, path, "_raw")

            if len(tar_lab_seen) == config.output_size*config.output_size:
                break
                

def export_clustering(model, node_name, test_set, processing, config, default=True, n_components=5):
    """
    Args:
        model: a tf model
        test_set: generator of (x, y), no_event
        processing: function to preprocess test_set x, y
        config: module config
        extractor: takes a dict of dict
        n_components: perform pca before clustering (no flattening of norm)
    """
    # get index in features of eta, phi and dep
    eta_idx = phi_idx = dep_idx = e_density_idx = None
    for idx, mode in enumerate(config.modes):
        if mode == "eta":
            eta_idx = idx
        if mode == "phi":
            phi_idx = idx
        if mode == "dep":
            dep_idx = idx
        if mode == "e_density":
            e_density_idx = idx

    tar_lab_seen = set()
    for (x, y), _ in test_set:
        # np array, truth nparticles, predicted nparticles
        node_eval, tar, lab = model.eval_node(x, y, node_name, processing)
        tar, lab = tar[0] + 1, lab[0] + 1

        if (tar, lab) not in tar_lab_seen:
            print "- extracting layers for true label {}, pred {} in {}".format(
                                            tar, lab, config.plot_output)

            tar_lab_seen.add((tar, lab))
            # reduce dimensionality
            pca = PCA(n_components=5)
            pca.fit(node_eval)
            # compute kmeans
            kmeans = KMeans(init='k-means++', n_clusters=int(lab), n_init=10)
            clusters = kmeans.fit_predict(node_eval)

            # recreate cells for plotting
            cells = dict()
            for i, (v, cluster) in enumerate(zip(x, clusters)):
                cells[i] = {"eta": v[eta_idx],
                            "phi": v[phi_idx],
                            "dep": v[dep_idx],
                            "e_density": v[e_density_idx],
                            "cluster": cluster + 1}
            eta_center = np.mean([c["eta"] for i, c in cells.iteritems()])
            phi_center = np.mean([c["phi"] for i, c in cells.iteritems()])

            if default:
                extractor = config.extractor
                extractor.modes = ["e_density", "cluster"]
                extractor.filter_mode = "e_density"
            
            else:
                extractor = get_custom_extractor(cells, eta_center, phi_center)

            matrices = extractor(cells, eta_center, phi_center)
            extractor.generate_report()

            path = config.plot_output + "true_{}_pred{}/".format(tar, lab)   
            check_dir(path)
            export_matrices(matrices, path, vmax=config.output_size+1)

        if len(tar_lab_seen) == config.output_size*config.output_size:
            break

def get_min_delta(l):
    ll = []
    for i, e in enumerate(l):
        for j in range(i, len(l)):
            if (abs(e - l[j])) > 0.01:
                ll += [abs(e - l[j])]
    if len(ll) == 0:
        return 0.1

    return min(min(ll), 0.1)

def get_min_deltaR(l):
    ll = []

    for i, e in enumerate(l):
        for j in range(i, len(l)):
            d = deltaR(e, l[j])
            if d > 0.001:
                ll += [d]

    if len(ll) == 0:
        return 0.1

    return min(min(ll), 0.1)

def get_min_deltas(l):
    l0 = []
    l1 = []

    for i, e in enumerate(l):
        for j in range(i, len(l)):
            d = deltaR(e, l[j])
            deta = abs(e[0] - l[j][0])
            dphi = abs(e[1] - l[j][1])
            if deta > 0.5*d and deta > 0.001:
                l0 += [deta]
            if dphi > 0.5*d and dphi > 0.001:
                l1 += [dphi]

    if len(l0) == 0:
        deta = 0.1
    else:
        deta = min(l0)
    if len(l1) == 0:
        dphi = 0.1
    else:
        dphi = min(l1)

    return deta, dphi


def deltaR(e, f):
    return np.sqrt((e[0]-f[0])**2 + (e[1]-f[1])**2)


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


def outputConfusionMatrix(tar, lab, part_min, output_size, filename):
    """ Generate a confusion matrix """
    cm = confusion_matrix(tar, lab, labels=range(output_size))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    classes = range(part_min, part_min + output_size)
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

def outputPerfProp(tar, lab, lead_props, filename, bins=8, av="macro", output_size=2, eval_perf_class=None, part_min=1):
    tars = defaultdict(list)
    labs = defaultdict(list)
    f1s = [0]*bins
    for t, l, p in zip(tar, lab, lead_props):
        idx = min(int(p*bins), bins-1)
        if eval_perf_class is None:
            tars[idx] += [t]
            labs[idx] += [l]
        else:
            if t == eval_perf_class:
                tars[idx] += [t]
                labs[idx] += [l]

    for b in range(bins):
        if eval_perf_class is None:
            f1 = f1_score(tars[b], labs[b], labels=range(output_size), average=av)
        else:
            f1 = sum(map(lambda (x, y): x == y, zip(tars[b], labs[b]))) / float(max(len(tars[b]), 1))
        f1s[b] = f1

    plt.figure()
    plt.plot(map(lambda x: x/float(bins) + 1./float(2*bins),range(bins)), f1s)
    x_lab = " - cluster with {} parts".format(eval_perf_class+part_min) if eval_perf_class is not None else ""
    plt.xlabel('Fraction of the leading particle{}'.format(x_lab))
    plt.ylabel('F1 score - {}'.format(av))
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
        lab_toprint = "lab {}".format(lab + config.part_min) + " "*(len(max(averages, key=len)) - len("lab {}".format(lab)))
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


