import ROOT
import numpy as np


def get_leadjet(entry):
    """
    Returns a ROOT 4d vector corresponding to the leadjet,
    the jet with highest pT
    (The leading two such jets should always have mass near 80 or 90 GeV)

    Args:
        entry: an entry of a ROOT tree
    Returns:
        leadjet: 4d ROOT vector (pT, eta, phi, PM) 
            pT : transverse momentum
            PM : mass of the truth jets
    """
    leadjet = ROOT.TLorentzVector(1,2,3,4)

    for j in range(entry.NJets):
        if (entry.JetsPt[j] > leadjet.Pt()):
            leadjet.SetPtEtaPhiM(entry.JetsPt[j],entry.JetsEta[j],entry.JetsPhi[j],entry.JetsPM[j])
            pass
        pass

    return leadjet

def get_truth_parts(mytree):
    """
    Returns a dict with information about truth particles

    Args:
        mytree: a ROOT tree
    Returns:
        truth_parts: a dict d[barcode] = dict({"pt": pt ...})
    """
    truth_parts = dict()
    for j in range(mytree.Truth_N):
        truth_barcode = mytree.Truth_barcode[j]
        truth_pt = mytree.Truth_Pt[j]
        truth_eta = mytree.Truth_Eta[j]
        truth_phi = mytree.Truth_Phi[j]
        truth_parts[truth_barcode] = {"pt": truth_pt, "eta": truth_eta, "phi": truth_phi}
        # print truth_barcode, truth_pt, truth_eta, truth_phi

    return truth_parts


def get_cells(mytree):
    """
    Returns a dict with information about cells
    Args:
        mytree: an entry of a ROOT tree

    Returns: 
        cells: dict = {"eta": eta, ...}
    """
    cells = dict()
    for j in range(mytree.Cell_N):
        uid = mytree.Cell_ID[j] # unique id of cell
        eta = mytree.Cell_eta[j] # eta
        phi = mytree.Cell_phi[j] # phi
        dep = mytree.Cell_dep[j] # cal layer
        e = mytree.Cell_E[j] # energy deposited in the cell
        vol = mytree.Cell_vol[j] # volume
        
        barcodes = mytree.Cell_barcodes
        # print len(barcodes) # prints 0
        # print mytree.Cell_barcodes
        # print("{}, {}, {}, {}".format(cell_eta, cell_phi, cell_dep, cell_vol))
        cells[uid] = {"eta": eta, "phi": phi, 
             "dep": dep, "e": e, "vol": vol}

    return cells

def get_tracks(mytree):
    """
    Returns a dict with information about the tracks
    Args:
        mytree: ROOT tree
    Returns:
        tracks: dict s.t. tracks[track_id] = {"eta": ..., "phi": ...}
    """
    nb_tracks = mytree.Track_N
    nb_truth_parts = mytree.Truth_N

    tracks = dict()
    for j in range(nb_tracks):
        pt = mytree.Track_Pt[j]
        eta = mytree.Track_Eta[j]
        phi = mytree.Track_Phi[j]
        barcode = mytree.Track_barcode[j]
        # Track_MCprob = mytree.Track_MCprob[j] # TODO, what is it?
        uid = mytree.Track_ID[j]
        tracks[uid] = {"eta": eta, "phi": phi, "barcode": barcode}

    return tracks

def topo_cluster_in_jet(leadjet, mytree, j):
    """
    Returns true if the j-th topocluster for mytree is 
    within 0.4 of the leadjet vector

    Args:
        leadjet: ROOT 4d vector
        mytree: ROOT tree
        j: int, index of topocluster

    Returns:
        boolean
    """
    topovec = ROOT.TLorentzVector()
    topovec.SetPtEtaPhiM(mytree.Topocluster_E[j]/np.cosh(mytree.Topocluster_eta[j]),
        mytree.Topocluster_eta[j],mytree.Topocluster_phi[j],0.)
    # only take the clusters inside the highest pT jet!
    return leadjet.DeltaR(topovec) > 0.4


def map_cells(cells, cell_ids, cell_weights=None):
    """
    Extracts cell information from cells dict for the cells whose id
    is in cell_ids.

    Args:
        cells: dict of dict.
            cells[id] = {"eta": ..., "phi": ... etc.}
        cell_ids: list of id (int) of cells
        cell_weights: (optional) list of weights, same len as cell_ids,
            weights of each cell in the
    Returns:
        
    """
    cells_eta = [cells[cell_id]["eta"] for cell_id in cell_ids]
    cells_phi = [cells[cell_id]["phi"] for cell_id in cell_ids]
    cells_dep = [cells[cell_id]["dep"] for cell_id in cell_ids]
    cells_e = [cells[cell_id]["e"] for cell_id in cell_ids]
    cells_vol = [cells[cell_id]["vol"] for cell_id in cell_ids]
    range_eta = max(cells_eta) - min(cells_eta)
    range_phi = max(cells_phi) - min(cells_phi)
    range_dep = max(cells_dep) - min(cells_dep)
    vol_tot = sum(cells_vol)
    e_tot = sum([e * w for e, w in zip(cells_e, cell_weights)] if cell_weights is not None else cells_e)

    return range_eta, range_phi, range_dep, vol_tot, e_tot

def map_truth_parts(truth_parts, mytree, j):
    """
    Maps barcodes of the j-th topocluster particles to their data
    in truth_parts dictionary

    Args:
        truth_parts: a dict d[barcode] = dict({"pt": pt ...}) 
            with data about truth particles
        mytree: a ROOT tree
        j: (int), j-th entry of the tree

    Returns:
        Extracted information from truth_parts
    """
    # pdgids and barcodes (same length)
    pdgids = mytree.Topocluster_pdgids[j] # type of particle?
    barcodes = mytree.Topocluster_barcodes[j] # id of the particle
    
    # truth_pt = [uid_ for uid_ in barcodes if uid_ not in truth_parts]
    # TODO: issue, sometimes there is uid_ = 0 in barcodes that as no 
    # correspondance in the truth_parts barcode extracted for the event.
    truth_pt = [uid_ for uid_ in barcodes if uid_ in truth_parts]
    # print truth_pt

    return True

def nb_of_truth_parts(mytree, j):
    """
    Returns the nb of truth particles that deposit more than 
    0.1 of their energy in the topocluster

    Args:
        mytree: Root tree
        j: index of tree entry

    Returns:
        nparts: (int) nb of truth particles that deposited > 0.1
        len(...): (int) nb of truth particles in the clusters
        props: list of the proportion, to plot
    """
    nparts = 0
    props = []
    for k in range(len(mytree.Topocluster_truthEfrac[j])):
        prop = mytree.Topocluster_truthEfrac[j][k]
        props += [prop]
        if (prop > 0.1):
            nparts+=1
            pass
        pass

    return nparts, len(mytree.Topocluster_truthEfrac[j]), props

