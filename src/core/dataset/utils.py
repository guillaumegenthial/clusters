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
            leadjet.SetPtEtaPhiM(entry.JetsPt[j], entry.JetsEta[j],
                                 entry.JetsPhi[j], entry.JetsPM[j])
            pass
        pass

    # print leadjet[0], leadjet[1], leadjet[2], leadjet[3]
    return leadjet

def get_leadjets(entry, min_pt=20, max_pt=2000, min_eta=0, max_eta=1):
    """
    Returns a list of ROOT 4d vector corresponding to the leadjets,

    Args:
        entry: an entry of a ROOT tree
        min_energy: min energy of jet in GeV
        max_eta: max eta range
    Returns:
        leadjets: a list of 4d ROOT vector (pT, eta, phi, PM) 
            pT : transverse momentum
            PM : mass of the truth jets
    """
    leadjets = []
    for j in range(entry.NJets):
        if ((max_pt > entry.JetsPt[j] > min_pt) and 
            (min_eta < abs(entry.JetsEta[j]) < max_eta)):
        
            jet = ROOT.TLorentzVector()
            jet.SetPtEtaPhiM(entry.JetsPt[j], entry.JetsEta[j],
                             entry.JetsPhi[j], entry.JetsPM[j])
            leadjets += [jet]
            pass
        pass

    return leadjets


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
        track_id = mytree.Truth_MatchTrack_ID[j]
        truth_parts[truth_barcode] = {"pt": truth_pt, "eta": truth_eta, 
                    "phi": truth_phi, "track": track_id}

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
        cells[uid] = {"eta": eta, "phi": phi,  "dep": dep, "e": e, "vol": vol}

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
        True if topo cluster in a 0.4 of leadjet
    """
    topovec  = ROOT.TLorentzVector()
    topo_eta = mytree.Topocluster_eta[j]
    topo_phi = mytree.Topocluster_phi[j]
    topo_pt  = mytree.Topocluster_E[j]/np.cosh(mytree.Topocluster_eta[j])
    topovec.SetPtEtaPhiM(topo_pt, topo_eta, topo_phi, 0.)
    # only take the clusters inside the highest pT jet!
    return leadjet.DeltaR(topovec) < 0.4, (topo_pt, topo_eta, topo_phi)

def topo_cluster_in_jets(leadjets, mytree, j):
    """
    Returns true if the j-th topocluster for mytree is 
    within 0.4 of one of the leadjets

    Args:
        leadjets: a list of ROOT 4d vector
        mytree: ROOT tree
        j: int, index of topocluster

    Returns:
        boolean
    """
    for leadjet in leadjets:
        is_in_jet, jet = topo_cluster_in_jet(leadjet, mytree, j)
        if is_in_jet:
            return jet
    # default
    return (0, 0, 0)

def map_cells(cells, cell_ids):
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
        dictionary: {cell_id: {"eta": eta, ...}, ...} 
    """
    return {id_: d_ for id_, d_ in cells.iteritems() if id_ in cell_ids}

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
    
    # TODO: issue, sometimes there is uid_ = 0 in barcodes that as no 
    # correspondance in the truth_parts barcode extracted for the event.
    # no_match = [uid_ for uid_ in barcodes if uid_ not in truth_parts]
    # if len(no_match) > 0:
    #     print "Parts not found: ", no_match
    
    return {id_: d_ for id_, d_ in truth_parts.iteritems() if id_ in barcodes}

def nb_of_truth_parts(mytree, j):
    """
    Returns the nb of truth particles that deposit more than 
    0.1 of their energy in the topocluster

    Args:
        mytree: Root tree
        j: index of tree entry

    Returns:
        nparts: (int) nb of truth particles that deposited > 0.1
    """
    nparts = 0
    props = []
    for k in range(len(mytree.Topocluster_truthEfrac[j])):
        prop = mytree.Topocluster_truthEfrac[j][k]
        if (prop > 0.1):
            nparts+=1
            props += [prop]

    return nparts, sorted(props, reverse=True)

def full_nb_of_truth_parts(mytree, j):
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


