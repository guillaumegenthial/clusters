import ROOT
import numpy as np
from utils import get_leadjet, get_my_print, \
    get_cells, get_truth_parts, topo_cluster_in_jet, \
    map_cells, nb_of_truth_parts, map_truth_parts, \
    get_tracks
from ROOT_plots_wrapper import Hist, Plots
import config

my_print = get_my_print(config.VERBOSE)

# 0. ROOT data
myfile = ROOT.TFile("data/ntuple_v3_2000k.root")
mytree = myfile.Get("SimpleJet")

# 1. ROOT histograms
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
plots.add("CellWeights", "Weights of cells inside a cluster", 20, -0.1, 4)

data = {"input": [], "output": []}
# 2. iterate over entries
for i in range(min(mytree.GetEntries(), config.MAX_EVENTS)):
    mytree.GetEntry(i)
    plots.fill("NJets", mytree.NJets)
    if (i%100==0):
        my_print("Entry {}, NJets {}".format(i, mytree.NJets))

    leadjet     = get_leadjet(mytree)
    cells       = get_cells(mytree)
    truth_parts = get_truth_parts(mytree)
    tracks      = get_tracks(mytree)

    for j in range(len(mytree.Topocluster_E)):
        if (topo_cluster_in_jet(leadjet, mytree, j)): 
            continue

        cell_ids = [id_ for id_ in mytree.Topocluster_cellIDs[j]]
        cell_weights = [w_ for w_ in mytree.Topocluster_cellWeights[j]]

        range_eta, range_phi, range_dep, vol_tot, e_tot = map_cells(cells, cell_ids, cell_weights)
        nparts, nparts_tot, props = nb_of_truth_parts(mytree, j)
        _ = map_truth_parts(truth_parts, mytree, j)

        data["input"].append([range_eta, range_phi, range_dep, vol_tot, e_tot])
        data["output"].append(nparts)

        plots.fills("CellWeights", cell_weights)
        plots.fills("TruthEfrac", props)
        plots.fill("NCells", len(cell_ids))
        plots.fill("DEta", range_eta)
        plots.fill("DPhi", range_phi)
        plots.fill("Dep", range_dep)
        plots.fill("Voltot", vol_tot)
        plots.fill("Etot", e_tot)
        plots.fill("NTruth_tot", nparts_tot)
        plots.fill("NTruth_filtered", nparts)

        my_print("Cluster: {} cells, {} particles. Ranges: eta {}, phi {}".format(
                len(cell_ids), nparts, range_eta, range_phi), 0)


# 4. Save histogram
suffix = "_"+config.EXP_NAME
plots.export(d="plots", suffix=suffix)
