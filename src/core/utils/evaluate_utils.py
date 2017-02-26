import copy
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from features_utils import Extractor
from general_utils import check_dir


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
                plt.savefig(path+"layer_{}{}.png".format(i_, suffix))
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

