import ROOT
import numpy as np
from utils import get_lead_jet, get_my_print, \
    get_cells_info, get_truth_parts, topo_cluster_in_jet, \
    extract_cells, nb_of_truth_parts
from ROOT_plots_wrapper import Hist, Plots
import config

my_print = get_my_print(config.VERBOSE)

# 0. ROOT data
myfile = ROOT.TFile("data/ntuple_v3_2000k.root")
mytree = myfile.Get("SimpleJet")


# # 1. ROOT histograms
plots = Plots()
plots.add("NJets", "Truth jet multiplicity")
plots.add('NCells', "Number of cells per cluster", 20, 0.5, 1000)
plots.add('DPhi', "Range of phi inside a cluster", 20,-0.5,2.5)
plots.add('DEta', "Range of eta inside a cluster", 20,-0.5,2.5)
plots.add("Dep", "Range of dep inside a cluster", 20, -0.5, 10)
plots.add('NTruth_filtered', "Number of truth particles contributing to a cluster > 0.1", 20,-0.5,15)
plots.add('NTruth_tot', "Number of truth particles contributing to a cluster tot", 20,-0.5,100)
plots.add("TruthEfrac", "Proportion of deposit of truth particle", 20, -0.1, 0.5)
plots.add("Etot", "Sum of cell energies in a cluster", 20, -100, 100)
plots.add("Voltot", "Volume of the cells in a cluster", 20, 0, 100)
plots.add("CellWeights", "Weights of cells inside a cluster", 20, -0.1, 1.1)


# 2. iterate over entries
for i in range(min(mytree.GetEntries(), config.MAX_EVENTS)):
    mytree.GetEntry(i)
    plots.fill("NJets", mytree.NJets)
    if (i%100==0):
        my_print("Entry {}, NJets {}".format(i, mytree.NJets))

    leadjet = get_lead_jet(mytree)
    cells = get_cells_info(mytree)
    my_print("Nb of truth particles {}".format(mytree.Truth_N), 0)
    truth_parts = get_truth_parts(mytree)

    for j in range(len(mytree.Topocluster_E)):
        if (topo_cluster_in_jet(leadjet, mytree, j)): 
            continue

        cell_ids = [id_ for id_ in mytree.Topocluster_cellIDs[j]]
        cell_weights = [w_ for w_ in mytree.Topocluster_cellWeights[j]]
        plots.fills("CellWeights", cell_weights)
        range_eta, range_phi, range_dep, vol_tot, e_tot = extract_cells(cells, cell_ids, cell_weights)
        my_print("sum: %.4fGev, cluster: %.4fGeV" % (e_tot/1000, mytree.Topocluster_E[j]), 1)

        plots.fill("NCells", len(cell_ids))
        plots.fill("DEta", range_eta)
        plots.fill("DPhi", range_phi)
        plots.fill("Dep", range_dep)
        plots.fill("Voltot", vol_tot)
        plots.fill("Etot", e_tot)

        nparts, nparts_tot, props = nb_of_truth_parts(mytree, j)
        plots.fills("TruthEfrac", props)
        plots.fill("NTruth_tot", nparts_tot)
        plots.fill("NTruth_filtered", nparts)

        # pdgids and barcodes (same length)
        pdgids = mytree.Topocluster_pdgids[j]
        barcodes = mytree.Topocluster_barcodes[j]
        
        truth_pt = [truth_parts[uid_] for uid_ in barcodes if uid_ in truth_parts]
        # print len(truth_pt), len(barcodes)
        # print truth_parts.keys()
        # print truth_pt
        
        my_print("Cluster: {} cells, {} particles. Ranges: eta {}, phi {}".format(
                len(cell_ids), nparts, range_eta, range_phi), 0)


# 4. Save histogram
plots.export()
