import ROOT
import numpy as np
from utils.data_utils import get_leadjet, get_leadjets, \
    get_cells, get_truth_parts, topo_cluster_in_jet, \
    map_cells, nb_of_truth_parts, map_truth_parts, \
    get_tracks, simple_features, full_nb_of_truth_parts, \
    topo_cluster_in_jets
from utils.general_utils import get_my_print
from ROOT_plots_wrapper import Hist, Plots
import config

# 0. ROOT data
myfile = ROOT.TFile("data/ntuple_v3_2000k.root")
mytree = myfile.Get("SimpleJet")

# 1. ROOT histograms
canvas = ROOT.TCanvas('draw', 'draw', 0, 0, 800, 800)
canvas.cd()
# plots = Plots()
# plots.add("Frac", "Dist of contribution to cluster", 1000, -0.01, 0.01)
# plots.add("Nparts", "Dist of contribution to cluster", 11, 0, 10)


# 2. iterate over entries
for i in range(min(mytree.GetEntries(), config.max_events)):
    mytree.GetEntry(i)
    leadjet     = get_leadjet(mytree)
    leadjets    = get_leadjets(mytree)
    cells       = get_cells(mytree)
    truth_parts = get_truth_parts(mytree)
    tracks      = get_tracks(mytree)

    max_barcode = max(truth_parts.keys())
    # print max_barcode

    for id_, d_ in tracks.iteritems():
        barcode = d_["barcode"]
        if barcode == max_barcode:
            print "Found"
        # eta_track = d_["eta"]
        # phi_track = d_["phi"]
        # if barcode == 0:
        #     for bc_, p_ in truth_parts.iteritems():
        #         eta_, phi_ = p_["eta"], p_["phi"]
        #         if abs(eta_track - eta_) < 0.1 and abs(phi_track - phi_) < 0.1:
        #             print "FOUND ", bc_, len(truth_parts)
        #             print "eta ", eta_track, eta_
        #             print "phi ", phi_track, phi_

        # eta_part = truth_parts[barcode]["eta"]
        # phi_part = truth_parts[barcode]["phi"]
        # print "eta ", eta_track, eta_part
        # print "phi", phi_track, phi_part


    print("Entry {}, NJets {}, NLeadJets {}".format(i, 
                        mytree.NJets, len(leadjets)))

    for j in range(len(mytree.Topocluster_E)):
        if not (topo_cluster_in_jet(leadjet, mytree, j)): 
            continue

        topo_eta = mytree.Topocluster_eta[j]
        topo_phi = mytree.Topocluster_phi[j]

        cell_ids = [id_ for id_ in mytree.Topocluster_cellIDs[j]]
        cell_weights = [w_ for w_ in mytree.Topocluster_cellWeights[j]]

        topo_cells = map_cells(cells, cell_ids)
        F1 = ROOT.TH2F("layer", "title", 40, -0.2, 0.2, 40, -0.2, 0.2)
        # F3 = ROOT.TH3F("layer", "title", 8, -0.2, 0.2, 8, -0.2, 0.2, 1, 1, 6)
        npoints = 0
        for id_, d_ in topo_cells.iteritems():
            eta, phi, e, dep = d_["eta"], d_["phi"], d_["e"], d_["dep"]
            # F3.Fill(eta - topo_eta, phi - topo_phi, dep)
            if dep == 1:
                npoints += 1
                F1.Fill(eta - topo_eta, phi - topo_phi, e)

        # F.DrawClone("Cont1");
        F1.DrawClone("Colz")
        # F3.DrawClone("BOX")
        # F.DrawClone("Colz");
        # F.DrawClone("lego2");
        # F.DrawClone("surf3");



        wait = input("PRESS ENTER TO CONTINUE.")

        topo_parts = map_truth_parts(truth_parts, mytree, j)

        topovec = ROOT.TLorentzVector(mytree.Topocluster_E[j]/np.cosh(mytree.Topocluster_eta[j]),
            mytree.Topocluster_eta[j],mytree.Topocluster_phi[j],0.)

        nparts_ = 0
        for _, p_ in topo_parts.iteritems():
            pt, eta, phi = p_["pt"], p_["eta"], p_["phi"]
            partvec = ROOT.TLorentzVector(pt, eta, phi,0.)
            if topovec.DeltaR(partvec) < 1:
                nparts_ += 1

        features = simple_features(topo_cells)
        nparts, nparts_tot, props = full_nb_of_truth_parts(mytree, j)
        # plots.fills("Frac", props)
        # plots.fill("Nparts", nparts)

        print("Cluster: {} cells. Parts: {} event, {} - {} cluster, {} > 0.1, {} DR".format(
                len(cell_ids), len(truth_parts), nparts_tot, len(topo_parts), nparts, nparts_))


# 4. Save histogram
# suffix = "_"+config.exp_name
# plots.export(d="plots", suffix=suffix)
