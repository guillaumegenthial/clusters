import ROOT
import numpy as np
from utils.data_utils import get_leadjet, get_leadjets, \
    get_cells, get_truth_parts, topo_cluster_in_jet, \
    map_cells, nb_of_truth_parts, map_truth_parts, \
    get_tracks, full_nb_of_truth_parts, \
    topo_cluster_in_jets
from utils.general_utils import get_my_print, \
    export_matrices
from ROOT_plots_wrapper import Hist, Plots
import config
from collections import Counter
from utils.features_utils import Extractor, simple_features
# 0. ROOT data
myfile = ROOT.TFile("data/ntuple_v3_2000k.root")
mytree = myfile.Get("SimpleJet")

# 1. ROOT histograms
canvas = ROOT.TCanvas('draw', 'draw', 0, 0, 800, 800)
canvas.cd()
plots = Plots()
# plots.add("Frac", "Dist of contribution to cluster", 1000, -0.01, 0.01)
# plots.add("Nparts", "Dist of contribution to cluster", 11, 0, 10)
plots.add("Deps", "Deps", 21, 0, 20)
plots.add("r_eta", "r_eta", 50, 0, 3)
plots.add("r_phi", "r_hi", 50, 0, 3)
plots.add("e_min", "e_min", 50, -1000, 10000)
plots.add("e_max", "e_max", 50, 0, 300000)
geom_cells = dict()
deps = dict()
for i in range(30):
    deps[i] = {"eta": [], "phi": [], "eta_max": 0, "eta_min": 10, "vol": set()}


plotted = False
# Extractor
extractor = Extractor(config.layer_extractors)

# 2. iterate over entries
for i in range(min(mytree.GetEntries(), 10)):
    mytree.GetEntry(i)
    leadjet     = get_leadjet(mytree)
    leadjets    = get_leadjets(mytree, min_energy=20, max_eta=config.max_eta)
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
        if not (topo_cluster_in_jets(leadjets, mytree, j)):
            continue

        topo_eta = mytree.Topocluster_eta[j]
        topo_phi = mytree.Topocluster_phi[j]

        # if not (abs(topo_eta) < 0.4):
        #     continue

        cell_ids = [id_ for id_ in mytree.Topocluster_cellIDs[j]]
        cell_weights = [w_ for w_ in mytree.Topocluster_cellWeights[j]]

        topo_cells = map_cells(cells, cell_ids)
        matrices = extractor(topo_cells, topo_eta, topo_phi)
        # print extractor.get_preservation_ratio(0)
        # export_matrices(matrices)

        # wait = input("Wait")

        # for i in range(23):
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     plt.imshow(matrices[i], interpolation='nearest', cmap="bwr",  vmin=-100, vmax=100)
        #     plt.colorbar()
        #     plt.grid(True)
        #     plt.savefig("plots/vizual_{}.png".format(i))
        # wait = input("PRESS ENTER TO CONTINUE.")

        # # F1 = ROOT.TH2F("layer", "title", 40, -0.2, 0.2, 40, -0.2, 0.2)
        # # F3 = ROOT.TH3F("layer", "title", 8, -0.2, 0.2, 8, -0.2, 0.2, 1, 1, 6)
        # npoints = 0
        # r_eta = 1.5 # data show that max eta range of a cluster is 2
        # d_eta = 0.025 # data show that this is the min diff between 2 etas
        # r_phi = 2 # data show ...
        # d_phi = 0.08 # from data
        # n_eta = int(r_eta/d_eta)
        # n_phi = int(r_phi/d_phi)
        # L0 = np.zeros([n_phi, n_eta])
        # non_zeros = 0
        # for id_, d_ in topo_cells.iteritems():
        #     eta, phi, e, dep, vol = d_["eta"], d_["phi"], d_["e"], d_["dep"], d_["vol"]
        #     deps[dep]["eta"] += [eta]
        #     deps[dep]["phi"] += [phi]
        #     deps[dep]["eta_max"] = max(abs(eta), deps[dep]["eta_max"])
        #     deps[dep]["eta_min"] = min(abs(eta), deps[dep]["eta_min"])
        #     deps[dep]["vol"].update([vol])
        #     plots.fill("Deps", dep)

        #     if dep == 0:
        #         non_zeros += 1
        #         eta_id = min(max(int((r_eta/2 + eta - topo_eta)/d_eta), 0), n_eta-1)
        #         print eta, topo_eta, eta - topo_eta,  eta_id, n_eta
        #         phi_id = min(max(int((r_phi/2 + phi - topo_phi)/d_phi), 0), n_phi-1)
        #         print phi, topo_eta, phi - topo_phi,  phi_id, n_phi
        #         L0[phi_id, eta_id] += e
                
        #     # F3.Fill(eta - topo_eta, phi - topo_phi, dep)
        #     if dep == 1:
        #         npoints += 1
        #         # F1.Fill(eta - topo_eta, phi - topo_phi, e)
        # print non_zeros, np.count_nonzero(L0)
        
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # plt.imshow(L0, interpolation='nearest', cmap="bwr",  vmin=-100, vmax=100)
        # plt.colorbar()
        # plt.grid(True)
        # plt.savefig("plots/vizual.png")
        # wait = input("PRESS ENTER TO CONTINUE.")
        
        # # F.DrawClone("Cont1");
        # # F1.DrawClone("Colz")
        # # F3.DrawClone("BOX")
        # # F.DrawClone("Colz");
        # # F.DrawClone("lego2");
        # # F.DrawClone("surf3");



        # wait = input("PRESS ENTER TO CONTINUE.")

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
        r_eta = features[0]
        r_phi = features[1]
        plots.fill("r_eta", r_eta)
        plots.fill("r_phi", r_phi)

        e_max = max([c_["e"] for c_ in topo_cells.itervalues()])
        e_min = min([c_["e"] for c_ in topo_cells.itervalues()])
        # print e_min, e_max
        plots.fill("e_min", e_min)
        plots.fill("e_max", e_max)
        nparts, nparts_tot, props = full_nb_of_truth_parts(mytree, j)

        if not plotted and nparts == 5:
            export_matrices(matrices)
            print "exporting matrices in plots"
            plotted = True

        # plots.fills("Frac", props)
        # plots.fill("Nparts", nparts)

        # print("Cluster: {} cells. Parts: {} event, {} - {} cluster, {} > 0.1, {} DR".format(
        #         len(cell_ids), len(truth_parts), nparts_tot, len(topo_parts), nparts, nparts_))


# print deps
# 4. Save histogram
suffix = "_"+config.exp_name
plots.export(d="plots", suffix=suffix)
extractor.generate_report()


# def get_min_delta(d):
#     d = list(set(d))
#     min_delta = 1
#     for i in range(len(d)):
#         for j in range(i):
#             delta = abs(d[i] - d[j])
#             if delta < min_delta and delta > 0.0025:
#                 min_delta = delta
#     return min_delta

# for i in xrange(30):
#     eM = deps[i]["eta_max"]
#     em = deps[i]["eta_min"]
#     de = get_min_delta(deps[i]["eta"])
#     dp = get_min_delta(deps[i]["phi"])
#     print "layer {:2d}, eta_m {:.4f}, eta_M {:.4f}, eta_r {:.6f}, phi_r {:.6f}".format(i, em, eM, de, dp)
#     print deps[i]["vol"]
# plots.add("Vol0", "Vol0", 20, 50000, 300000)
# plots.fills("Vol0", list(deps[0]["vol"]))
# 
