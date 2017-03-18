from core.dataset.root import DatasetRoot
from core.dataset.pickle import DatasetPickle

# params
max_events = 2000

# root generator - all
data = DatasetRoot(path="data/root/ntuple_v3_2000k.root", tree="SimpleJet", max_iter=max_events, 
    jet_filter=False, jet_min_pt=20, jet_max_pt=2000, jet_min_eta=0, jet_max_eta=1,  
    topo_filter=False, topo_min_pt=0, topo_max_pt=5, topo_min_eta=0, topo_max_eta=0.5)

# export
data.export_one_file_per_event("data/events2")
