import numpy as np

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


def matrix_extraction(cells, layer_extractors, topo_eta, topo_phi):
    """
    Returns matrices from cells
    Args:
        cells a dict of dict
        layer_extractors: a dict of layer extractor, one for each layer
        topo_eta: eta of cluster
        topo_phi: phi of cluster
    Returns
        a dict of np arrays d[layer nb] = np array
    """
    matrices = dict()
    for dep, extractor in layer_extractors.iteritems():
        assert dep == extractor.dep, "Dep mismatch in {}".format(
                                        self.__class__.__name__)
        matrices[dep] = np.zeros([extractor.n_phi, extractor.n_eta])

    for id_, cell_ in cells.iteritems():
        dep = cell_["dep"]
        eta_id, phi_id = layer_extractors[dep](cell_, topo_eta, topo_phi)
        matrices[dep][phi_id, eta_id] += cell_["e"]

    return matrices



