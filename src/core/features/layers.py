import numpy as np

def wrap_extractor(extractor):
    def f(d_):
        cells    = d_["topo_cells"]
        topo_eta = d_["topo_eta"]
        topo_phi = d_["topo_phi"]
        matrices = extractor(cells, topo_eta, topo_phi)
        return matrices

    return f

def extractor_default_preprocess(extractor):

    def f(data):
        print "preprocessing..."
        # prepare storage for statistics
        eps = 10^(-6)
        mean = dict()
        var = dict()
        counts = dict()
        for dep, l_ext in extractor.layer_extractors.iteritems():
                mean[dep] = dict()
                var[dep] = dict()
                counts[dep] = 0
                for mode in extractor.modes:
                    mean[dep][mode] = np.zeros([l_ext.n_phi, l_ext.n_eta])
                    var[dep][mode] = np.zeros([l_ext.n_phi, l_ext.n_eta])

        # get statistics
        for x, y in data:
            for dep, modes_ in x.iteritems():
                found = False
                for mode, dat in modes_.iteritems():
                    if type(dat) != tuple:
                        mean[dep][mode] += dat
                        var[dep][mode] += dat**2
                        found = True
                if found:
                    counts[dep] += 1

        # compute statistics
        for dep in mean.iterkeys():
            counts_dep = counts[dep]
            for mode in extractor.modes:
                mean[dep][mode] /= counts_dep
                var[dep][mode] /= counts_dep
                var[dep][mode] -= (mean[dep][mode])**2

        # apply statistics
        for i in range(len(data)):
            for dep, modes_ in data[i][0].iteritems():
                for mode, dat in modes_.iteritems():
                    if type(dat) != tuple:
                        data[i][0][dep][mode] = (dat - mean[dep][mode]) / (np.sqrt(var[dep][mode]) + eps)

        print "- done."
        return data

    return f


def extractor_post_process(extractor, output_size):
    def f(X, Y):
        result = []
        for x in X:
            result_ = []
            for dep, modes_ in x.iteritems():
                for mode, dat in modes_.iteritems():
                    if type(dat) != tuple:
                        result_.append(dat)
                    else:
                        n_phi = extractor.layer_extractors[dep].n_phi
                        n_eta = extractor.layer_extractors[dep].n_eta
                        result_.append(np.zeros([n_phi, n_eta]))

            result.append(result_)

        return np.transpose(np.asarray(result), (0, 3, 2, 1)), np.minimum(Y, output_size-1)

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
    def __init__(self, layer_extractors, modes=["e", "vol"]):
        self.layer_extractors = layer_extractors
        self.preservation_ratio = {dep: {"cell": 0, "mat": 0} 
                        for dep in layer_extractors.iterkeys()}
        self.modes = modes

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
                    matrices[dep][mode][phi_id, eta_id] += cell_[mode]

        for dep, modes_ in matrices.iteritems():
            self.preservation_ratio[dep]["mat"] += np.count_nonzero(modes_["e"])
            if np.sum(modes_["e"]) == 0:
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
            print "Layer {}, cell_count {}, mat_count {}, ratio {}".format(
                                  dep, counts["cell"], counts["mat"], ratio)




