import ROOT
import numpy as np
import time
from utils import get_leadjets, \
    get_cells, get_truth_parts, topo_cluster_in_jets, \
    map_cells, nb_of_truth_parts, map_truth_parts, \
    get_tracks
from core.utils.general import check_dir, pickle_dump, Progbar


class DatasetRoot(object):
    def __init__(self, path, tree="SimpleJet", max_iter=10, 
        jet_filter=True, jet_min_pt=20, jet_max_pt=2000, jet_min_eta=0, jet_max_eta=1,  
        topo_filter=False, topo_min_pt=0, topo_max_pt=5, topo_min_eta=0, topo_max_eta=0.5):

        # general
        self.path = path
        self.myfile = ROOT.TFile(path)
        self.mytree = self.myfile.Get(tree)
        self.max_iter = max_iter
        self.length = None

        # filteron jets
        self.jet_filter = jet_filter
        self.jet_min_pt = jet_min_pt
        self.jet_max_pt = jet_max_pt
        self.jet_min_eta = jet_min_eta
        self.jet_max_eta = jet_max_eta

        # filter on topocluster
        self.topo_filter = topo_filter
        self.topo_min_pt = topo_min_pt
        self.topo_max_pt = topo_max_pt
        self.topo_min_eta = topo_min_eta
        self.topo_max_eta = topo_max_eta

    def __iter__(self):
        """
        Iterate over topoclusters that are in jets that respect max_eta
        and min_energy conditions
        Returns:
            a dict{"topo_eta": ....}
        """
        mytree = self.mytree
        max_iter = self.max_iter
        nb_iter = min(mytree.GetEntries(), max_iter) if max_iter else mytree.GetEntries()

        for i in range(nb_iter):
            mytree.GetEntry(i)

            leadjets    = get_leadjets(mytree, min_pt=self.jet_min_pt, max_pt=self.jet_max_pt, 
                                            min_eta=self.jet_min_eta, max_eta=self.jet_max_eta)
            cells       = get_cells(mytree) # sloooow
            truth_parts = get_truth_parts(mytree) # fast
            tracks      = get_tracks(mytree) # fast

            for j in range(len(mytree.Topocluster_E)):
                jet_pt, jet_eta, jet_phi = (topo_cluster_in_jets(leadjets, mytree, j)) 
                if self.jet_filter and (jet_pt, jet_eta, jet_phi) == (0, 0, 0):
                    continue
                
                topo_eta = mytree.Topocluster_eta[j]
                topo_phi = mytree.Topocluster_phi[j]
                topo_pt  = (mytree.Topocluster_E[j]/np.cosh(mytree.Topocluster_eta[j]))

                if self.topo_filter:
                    if not ((self.topo_min_eta < topo_eta < self.topo_max_eta) and 
                            (self.topo_min_pt < topo_pt < self.topo_max_pt)):
                        continue

                cell_ids = [id_ for id_ in mytree.Topocluster_cellIDs[j]]
                nparts = nb_of_truth_parts(mytree, j)

                topo_cells = map_cells(cells, cell_ids)

                yield {"topo_pt": topo_pt, "topo_eta": topo_eta, "topo_phi": topo_phi, 
                       "jet_pt": jet_pt, "jet_eta": jet_eta, "jet_phi": jet_phi, 
                       "topo_cells": topo_cells, "nparts": nparts, }, i
                       
            del cells
            del truth_parts
            del tracks
            del leadjets


    def __len__(self):
        """
        Return number of iteration
        Iterate over self and increment length
        Returns:
            length
        """
        if self.length is None:
            length = 0
            for i, _ in enumerate(self):
                length += 1
            self.length = length

        return self.length


    def get_data(self):
        """
        Iterates over the data and stores everything in a list
        Returns:
            a list of dict
        """
        data = []
        for d_, _ in self:
            data.append(d_)
        return data

    def export_one_file_per_event(self, path):
        """
        Iterates over dataset and dumps one file for each event
        """
        prog = Progbar(target=self.max_iter)
        check_dir(path)
        event_no_ref = 0
        event = []
        for cluster, event_no in self:
            if event_no != event_no_ref:
                export_path = path+"/event_{}_nclusters_{}.npy".format(event_no_ref, len(event))
                pickle_dump(event, export_path, False)
                event_no_ref = event_no
                event = []
                prog.update(event_no+1)
            event += [cluster]



