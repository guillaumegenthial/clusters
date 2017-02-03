import ROOT
import numpy as np
import time
from utils.data_utils import get_leadjet, \
    get_cells, get_truth_parts, topo_cluster_in_jet, \
    map_cells, nb_of_truth_parts, map_truth_parts, \
    get_tracks
from utils.general_utils import get_my_print


class Dataset(object):
    def __init__(self, path="data/ntuple_v3_2000k.root", tree="SimpleJet", max_iter=10, max_nparts=10, verbose=0):
        self.path = path
        self.myfile = ROOT.TFile(path)
        self.mytree = self.myfile.Get(tree)
        self.max_iter = max_iter
        self.max_nparts = max_nparts
        self.my_print = get_my_print(verbose)
        self.length = None

    def __iter__(self):
        mytree = self.mytree
        max_iter = self.max_iter
        nb_iter = min(mytree.GetEntries(), max_iter) if max_iter else mytree.GetEntries()

        for i in range(nb_iter):
            mytree.GetEntry(i)

            if (i%100==0):
                self.my_print("Entry {}, NJets {}".format(i, mytree.NJets), 1)

            leadjet     = get_leadjet(mytree)
            cells       = get_cells(mytree) # sloooow
            truth_parts = get_truth_parts(mytree) # fast
            tracks      = get_tracks(mytree) # fast

            for j in range(len(mytree.Topocluster_E)):
                if (topo_cluster_in_jet(leadjet, mytree, j)): 
                    continue

                cell_ids = [id_ for id_ in mytree.Topocluster_cellIDs[j]]
                cell_weights = [w_ for w_ in mytree.Topocluster_cellWeights[j]]

                range_eta, range_phi, range_dep, vol_tot, e_tot = map_cells(cells, cell_ids, cell_weights)
                nparts, nparts_tot, props = nb_of_truth_parts(mytree, j)
                _ = map_truth_parts(truth_parts, mytree, j)

                x = [range_eta, range_phi, range_dep, vol_tot, e_tot]
                y = nparts if nparts < self.max_nparts else (self.max_nparts-1)

                yield x, y


    def __len__(self):
        if self.length is None:
            length = 0
            for i, _ in enumerate(self):
                length += 1
            self.length = length

        return self.length


    def get_data(self):
        data = []
        for i, (x, y) in enumerate(self):
            data.append([x, y])
        self.length = i
        return data

if __name__ == "__main__":
    data = Dataset()
    t0 = time.time()
    for x, y in data:
        # print(y)
        continue
    t1 = time.time()
    print("Time elapsed : {:.2f}".format(t1 - t0))

