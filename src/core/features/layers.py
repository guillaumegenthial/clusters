from core.utils.general import Progbar
from utils import get_mode
import numpy as np

def get_default_processing(data, extractor, processing_y, statistics="default"):
    """
    Compute statistics over data and outputs a function
    Args:
        data: (generator) of (x, y), i tuples
    Returns:
        function f(x, y) where x, y are batches of x and y
    """
    print "Creating processing function"
    # prepare storage for statistics
    eps = 10^(-6)
    mean = dict()
    var = dict()
    mean_lay = dict()
    var_lay = dict()
    mean_mode = dict()
    var_mode = dict()
    counts_dep = dict()
    for mode in extractor.modes:
        mean_mode[mode] = 0
        var_mode[mode] = 0
    for dep, l_ext in extractor.layer_extractors.iteritems():
            mean[dep] = dict()
            var[dep] = dict()
            mean_lay[dep] = dict()
            var_lay[dep] = dict()
            counts_dep[dep] = 0
            for mode in extractor.modes:
                mean[dep][mode] = np.zeros([l_ext.n_phi, l_ext.n_eta])
                var[dep][mode] = np.zeros([l_ext.n_phi, l_ext.n_eta])
                mean_lay[dep][mode] = 0
                var_lay[dep][mode] = 0

    # get statistics
    prog = Progbar(target=data.max_iter)
    n_examples = 0
    for (x, y, _), i in data:
        n_examples += 1
        prog.update(i)
        for dep, modes_ in x.iteritems():
            found = False
            for mode, dat in modes_.iteritems():
                if type(dat) != tuple:
                    mean[dep][mode] += dat
                    var[dep][mode] += dat**2
                    mean_lay[dep][mode] += np.mean(dat)
                    var_lay[dep][mode] += np.mean(dat**2)
                    mean_mode[mode] += np.mean(dat)
                    var_mode[mode] += np.mean(dat**2)
                    found = True
            if found:
                counts_dep[dep] += 1

    data.length = n_examples
    prog.update(i+1)

    # compute statistics
    for dep in mean.iterkeys():
        for mode in extractor.modes:
            mean[dep][mode] /= counts_dep[dep]
            var[dep][mode] /= counts_dep[dep]
            var[dep][mode] -= (mean[dep][mode])**2

            mean_lay[dep][mode] /= counts_dep[dep]
            var_lay[dep][mode] /= counts_dep[dep]
            var_lay[dep][mode] -= (mean_lay[dep][mode])**2

    total_dep_seen = sum(counts_dep.values())
    for mode in extractor.modes:
        mean_mode[mode] /= total_dep_seen
        var_mode[mode] /= total_dep_seen
        var_mode[mode] -= (mean_mode[mode])**2

    print "- done."

    def f(X, Y):
        result = []
        # apply statistics
        for x in X:
            result_ = []
            for dep, modes_ in x.iteritems():
                for mode, dat in modes_.iteritems():
                    if type(dat) != tuple:
                        mat_ = dat
                    else:
                        n_phi = extractor.layer_extractors[dep].n_phi
                        n_eta = extractor.layer_extractors[dep].n_eta
                        mat_  = (np.zeros([n_phi, n_eta]))

                    if statistics == "default":
                        mat_ = (mat_ - mean[dep][mode]) / (np.sqrt(var[dep][mode]) + eps)
                    elif statistics == "mean":
                        mat_ = (mat_ - mean[dep][mode])
                    elif statistics == "scale":
                        mat_ = (mat_) / (np.sqrt(var[dep][mode]) + eps)

                    if statistics == "layer_default":
                        mat_ = (mat_ - mean_lay[dep][mode]) / (np.sqrt(var_lay[dep][mode]) + eps)
                    elif statistics == "layer_mean":
                        mat_ = (mat_ - mean_lay[dep][mode])
                    elif statistics == "layer_scale":
                        mat_ = (mat_) / (np.sqrt(var_lay[dep][mode]) + eps)

                    if statistics == "mode_default":
                        mat_ = (mat_ - mean_mode[mode]) / (np.sqrt(var_mode[mode]) + eps)
                    elif statistics == "mode_mean":
                        mat_ = (mat_ - mean_mode[mode])
                    elif statistics == "mode_scale":
                        mat_ = (mat_) / (np.sqrt(var_mode[mode]) + eps)

                    result_ += [mat_]

            result.append(result_)

        return np.transpose(np.asarray(result), (0, 2, 3, 1)), processing_y(Y), None

    return f

def wrap_extractor(extractor):
    def f(d_):
        cells    = d_["topo_cells"]
        topo_eta = d_["topo_eta"]
        topo_phi = d_["topo_phi"]
        matrices = extractor(cells, topo_eta, topo_phi)
        return matrices

    return f


class LayerExtractor(object):
    def __init__(self, dep, r_eta, d_eta, r_phi, d_phi):
        self.dep = dep
        self.r_eta = r_eta
        self.r_phi = r_phi
        self.d_eta = d_eta
        self.d_phi = d_phi
        
        self.n_eta = int(r_eta/d_eta)
        self.n_phi = int(r_phi/d_phi)

    def __call__(self, cell, topo_eta, topo_phi):
        """
        Returns phi_id, eta_id
        Args:
            cell: dict cell["eta"], ..., must have same dep as self
            topo_eta: eta of cluster
            topo_phi: phi of cluster
        Returns:
            phi_id: id in the matrix of phi
            eta_id: id in the matrix of eta
        """
        assert cell["dep"] == self.dep, "Dep mismatch in {}".format(
                                            self.__class__.__name__)
        r_eta = self.r_eta
        r_phi = self.r_phi
        d_eta = self.d_eta
        d_phi = self.d_phi
        n_eta = self.n_eta
        n_phi = self.n_phi

        eta = cell["eta"]
        phi = cell["phi"]

        eta_id = min(max(int((r_eta/2 + eta - topo_eta)/d_eta), 0), n_eta-1)
        # print eta, topo_eta, eta - topo_eta,  eta_id, n_eta
        phi_id = min(max(int((r_phi/2 + phi - topo_phi)/d_phi), 0), n_phi-1)
        # print phi, topo_eta, phi - topo_phi,  phi_id, n_phi

        return eta_id, phi_id

class Extractor(object):
    def __init__(self, layer_extractors, modes=["e", "vol"], filter_mode="e"):
        self.layer_extractors = layer_extractors
        self.preservation_ratio = {dep: {"cell": 0, "mat": 0} 
                        for dep in layer_extractors.iterkeys()}
        self.modes = modes
        self.filter_mode = filter_mode

    def __call__(self, cells, topo_eta, topo_phi):
        """
        Returns matrices from cells
        Args:
            cells a dict of dict
            layer_extractors: a dict of layer extractor, one for each layer
            topo_eta: eta of cluster
            topo_phi: phi of cluster
        Returns
            a dict of np arrays d[layer nb] = np array if np array is not empty
                                            else shape of the np array
        """
        layer_extractors = self.layer_extractors
        matrices = dict()
        for dep, extractor in layer_extractors.iteritems():
            assert dep == extractor.dep, "Dep mismatch in {}".format(
                                            self.__class__.__name__)
            matrices[dep] = dict()
            for mode in self.modes:
                matrices[dep][mode] = np.zeros([extractor.n_phi, extractor.n_eta])

        for id_, cell_ in cells.iteritems():
            dep = cell_["dep"]
            # only add matrices for which we have an extractor
            if dep in self.layer_extractors.keys():
                self.preservation_ratio[dep]["cell"] += 1
                eta_id, phi_id = layer_extractors[dep](cell_, topo_eta, topo_phi)
                for mode in self.modes:
                    if mode != "cluster":
                        matrices[dep][mode][phi_id, eta_id] += get_mode(cell_, mode)
                    else:
                        matrices[dep][mode][phi_id, eta_id] = get_mode(cell_, mode)
        
        for dep, modes_ in matrices.iteritems():
            self.preservation_ratio[dep]["mat"] += np.count_nonzero(modes_[self.filter_mode])
            if np.sum(modes_[self.filter_mode]) == 0:
                for mode in self.modes:
                    modes_[mode] = modes_[mode].shape

        return matrices

    def get_preservation_ratio(self, dep):
        """
        Returns ratio of total number of cells seen versus
        number of cells mapped in the matrices for dep dep
        Args:
            dep: (int) id of layer depth
        """
        mat_count = self.preservation_ratio[dep]["mat"]
        cell_count = self.preservation_ratio[dep]["cell"]
        if cell_count == 0:
            return 0.0
        else:
            return  mat_count / float(cell_count)

    def generate_report(self):
        """
        Print summary of counts and ratio to measure how good 
        our mapping is
        """
        for dep, counts in self.preservation_ratio.iteritems():
            ratio = self.get_preservation_ratio(dep)
            if ratio != 0:
                print "Layer {}, cell_count {}, mat_count {}, ratio {}".format(
                                  dep, counts["cell"], counts["mat"], ratio)


def get_custom_extractor(cells, eta_center, phi_center):
    layer_extractors = dict()
    for l in range(24):
        etas = [c["eta"] for i, c in cells.iteritems() if c["dep"] == l]
        phis = [c["phi"] for i, c in cells.iteritems() if c["dep"] == l]
        d_eta, d_phi = get_min_deltas(zip(etas, phis))

        if len(etas) != 0:
            r_etas = 2*max([abs(e - eta_center) for e in etas])
            r_phis = 2*max([abs(e - phi_center) for e in phis])
        else:
            r_etas = r_phis = 1.5
        r_etas = max(r_etas, d_eta)
        r_phis = max(r_phis, d_phi)

        layer_extractors[l] = LayerExtractor(l, r_etas, d_eta, r_phis, d_phi)

    extractor = Extractor(layer_extractors, ["e_density", "cluster"], "e_density")




