from core.utils.general import pickle_load, get_all_files

class DatasetPickle(object):
    def __init__(self, path, max_iter=10, 
        jet_filter=True, jet_min_pt=20, jet_max_pt=2000, jet_min_eta=0, jet_max_eta=1,  
        topo_filter=False, topo_min_pt=0, topo_max_pt=5, topo_min_eta=0, topo_max_eta=0.5):

        # general
        self.path = path
        self.files = get_all_files(path)
        self.max_iter = max_iter

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
        for i, file in enumerate(self.files):
            clusters = pickle_load(self.path + "/" + file, verbose=False)
            for cluster in clusters:
                jet = (cluster["jet_pt"], cluster["jet_eta"], cluster["jet_phi"])
                if self.jet_filter:
                    if jet == (0, 0, 0):
                        continue
                    if not ((self.jet_min_pt < jet[0] < self.jet_max_pt) and
                        (self.jet_min_eta < jet[1] < self.jet_max_eta)):
                        continue
                    
                if self.topo_filter:
                    if not ((self.topo_min_pt < cluster["topo_pt"] < self.topo_max_pt) and
                        (self.topo_min_eta < cluster["topo_eta"] < self.topo_max_eta)):
                        continue

                yield cluster, i
