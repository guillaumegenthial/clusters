import numpy as np

def simple_features(tops=2, mode=1):

    def f(d_, cell_weights=None):
        """
        Returns a fixed size np array
        Args:
            d_ : dict {"topo_cells": ...,}
        Returns:
            f: np array [range_eta, ...]
        """
        topo_eta = d_["topo_eta"]
        topo_phi = d_["topo_phi"]

        cells = d_["topo_cells"]
        ncells = len(cells)

        if ncells > 0:

            # general information
            cells_eta = [d_["eta"] for d_ in cells.itervalues()]
            cells_phi = [d_["phi"] for d_ in cells.itervalues()]
            cells_dep = [d_["dep"] for d_ in cells.itervalues()]
            cells_e = [d_["e"] for d_ in cells.itervalues()]
            cells_vol = [d_["vol"] for d_ in cells.itervalues()]
            cells_pt = map_pT(cells_e, cells_eta)
            r_eta = max(cells_eta) - min(cells_eta)
            r_phi = max(cells_phi) - min(cells_phi)
            r_dep = max(cells_dep) - min(cells_dep)
            vol_tot = sum(cells_vol)


            # get highest e
            cells_e_sorted = sorted(((d_["eta"], d_["phi"], d_["e"]) for d_ in cells.itervalues()),
                                    reverse=True, key=lambda (a, b, c): c)
            top_cells = cells_e_sorted[:tops]
            top_e = [(eta - topo_eta, phi - topo_phi, e) for (eta, phi, e) 
                        in top_cells]

            d_eta = list(zip(*top_e)[0]) + [0]*(tops - len(top_e))
            d_phi = list(zip(*top_e)[1]) + [0]*(tops - len(top_e))
            d_e = list(zip(*top_e)[2]) + [0]*(tops - len(top_e))
            dR = map_deltaR(d_eta, d_phi)

            # get hightest pT
            cells_pt_sorted = sorted(((d_["eta"], d_["phi"], pT(d_["e"], d_["eta"])) for d_ in cells.itervalues()),
                                    reverse=True, key=lambda (a, b, c): c)
            top_cells = cells_pt_sorted[:tops]
            top_pt = [(eta - topo_eta, phi - topo_phi, e) for (eta, phi, e) 
                        in top_cells]

            d_eta_pt = list(zip(*top_pt)[0]) + [0]*(tops - len(top_pt))
            d_phi_pt = list(zip(*top_pt)[1]) + [0]*(tops - len(top_pt))
            d_pt = list(zip(*top_pt)[2]) + [0]*(tops - len(top_pt))
            dR_pt = map_deltaR(d_eta, d_phi)

            # get total of energy and pt
            e_tot = sum([e * w for e, w in zip(cells_e, cell_weights)] if cell_weights is not None else cells_e)
            pt_tot = sum([pt * w for pt, w in zip(cells_pt, cell_weights)] if cell_weights is not None else cells_pt)
        
        else:
            r_eta = r_phi = r_dep = vol_tot = e_tot = pt_tot = 0
            d_e = dR = d_pt = dR_pt = [0] * tops


        if mode == 1:
            result =  np.array([ncells, topo_eta, topo_phi, r_eta, r_phi, r_dep, vol_tot, e_tot] 
                            + d_e + dR)
        elif mode == 2:
            result = np.array([ncells, topo_eta, topo_phi, r_eta, r_phi, r_dep, vol_tot, pt_tot] 
                            + d_pt + dR_pt)
        elif mode == 3:
            result = np.array([ncells, topo_eta, topo_phi, r_eta, r_phi, r_dep, vol_tot, e_tot, pt_tot] 
                            + d_e + dR + d_pt + dR_pt)
        elif mode == 4:
            result = np.array([r_eta, r_phi, vol_tot, e_tot])
        else:
            print "Unknown mode {} for simple_features".format(mode)
            raise NotImplementedError

        return result

    feat = ["ncells", "topo_eta", "topo_phi", "r_eta", "r_phi", "r_dep", "vol_tot", "e_tot", "pt_tot"]
    feat += ["d_e_{}".format(i) for i in range(tops)]
    feat += ["d_R_{}".format(i) for i in range(tops)]
    feat += ["d_pt_{}".format(i) for i in range(tops)]
    feat += ["dR_pt_{}".format(i) for i in range(tops)]

    return f, feat

def pT(I, eta):
    """
    Computes transversal momentum from cell intensity I and eta of the cell
    """
    return I/np.cosh(eta)

def map_pT(Is, etas):
    return [pT(I, eta) for (I, eta) in zip(Is, etas)]

def deltaR(eta, phi):
    return np.sqrt(eta**2 + phi**2)

def map_deltaR(etas, phis):
    return [deltaR(e, p) for (e, p) in zip(etas, phis)]
