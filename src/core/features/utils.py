import numpy as np


def e_density(e, vol):
    if vol == 0:
        return 0.0
    else:
        return (float(e)/float(vol))

def pT(e, eta):
    """
    Computes transversal momentum from cell energy e and eta of the cell
    """
    if np.cosh(eta) != 0:
        return e/np.cosh(eta)
    else:
        return 0

def get_mode(cell, mode):
    """
    return a float corresponding to the mode from a cell
    Args:
        cell: dict with "e", "vol", "eta", ...etc
        mode: (string)
    Returns:
        float corresponding to the mode
    """
    if mode in ["e", "vol", "eta", "phi"]:
        return cell[mode]
    if mode == "e_density":
        return e_density(cell["e"], cell["vol"])
    if mode == "pT":
        return pT(cell["e"], cell["eta"])
    if mode == "dep":
        dep = int(cell["dep"])
        v = [0]*(dep) + [1] + [0]*(23-dep)
        return v
    else:
        print "ERROR: {} mode is unknown".format(mode)
        raise NotImplementedError

