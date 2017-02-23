import ROOT
import numpy as np
import time
from data_utils import get_leadjets, \
    get_cells, get_truth_parts, topo_cluster_in_jets, \
    map_cells, nb_of_truth_parts, map_truth_parts, \
    get_tracks

from general_utils import get_my_print


class Dataset(object):
    def __init__(self, path="data/ntuple_v3_2000k.root", tree="SimpleJet", 
                 max_iter=10, verbose=0, max_eta=1, min_energy=20):
        self.path = path
        self.myfile = ROOT.TFile(path)
        self.mytree = self.myfile.Get(tree)
        self.max_iter = max_iter
        self.my_print = get_my_print(verbose)
        self.length = None
        self.max_eta = max_eta
        self.min_energy = min_energy

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

            if (i%100==0):
                self.my_print("Entry {}, NJets {}".format(i, mytree.NJets), 1)

            leadjets    = get_leadjets(mytree, min_energy=self.min_energy, 
                                        max_eta=self.max_eta)
            cells       = get_cells(mytree) # sloooow
            truth_parts = get_truth_parts(mytree) # fast
            tracks      = get_tracks(mytree) # fast

            for j in range(len(mytree.Topocluster_E)):
                if (topo_cluster_in_jets(leadjets, mytree, j)): 
                    continue

                topo_eta = mytree.Topocluster_eta[j]
                topo_phi = mytree.Topocluster_phi[j]

                cell_ids = [id_ for id_ in mytree.Topocluster_cellIDs[j]]
                nparts = nb_of_truth_parts(mytree, j)

                topo_cells = map_cells(cells, cell_ids)

                yield {"topo_eta": topo_eta, "topo_phi": topo_phi, 
                       "topo_cells": topo_cells, "nparts": nparts}
                       
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
        for i, d_ in enumerate(self):
            data.append(d_)
        self.length = i
        return data

if __name__ == "__main__":
    data = Dataset()
    t0 = time.time()
    for d_ in data:
        # print(y)
        continue
    t1 = time.time()
    print("Time elapsed : {:.2f}".format(t1 - t0))

