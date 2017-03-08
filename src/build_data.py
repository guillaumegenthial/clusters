from core.dataset.root import DatasetRoot
from core.dataset.pickle import DatasetPickle

# params
max_events = 2000

# root generator - all
data = DatasetRoot(path="data/root/ntuple_v3_2000k.root", tree="SimpleJet", max_iter=max_events, 
    jet_filter=False, jet_min_pt=20, jet_max_pt=2000, jet_min_eta=0, jet_max_eta=1,  
    topo_filter=False, topo_min_pt=0, topo_max_pt=5, topo_min_eta=0, topo_max_eta=0.5)

# export
data.export_one_file_per_event("data/events")

# pickle generator from files - custom filtering
data = DatasetPickle("data/events", max_iter=max_events, shuffle=False, 
    jet_filter=False, jet_min_pt=10, jet_max_pt=2000, jet_min_eta=0, jet_max_eta=1,  
    topo_filter=True, topo_min_pt=0, topo_max_pt=1000, topo_min_eta=0, topo_max_eta=1)

# filtering info
counts = 0
last_i = 0
for cluster, i in data:
    if last_i != i:
        print i, counts+1
        counts = 0
        last_i = i
    counts += 1


