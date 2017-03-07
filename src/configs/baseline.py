import numpy as np
from core.features.layers import LayerExtractor
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool

# general
exp_name = "baseline"

# general data
data_path = "data/ntuple_v3_2000k.root"
max_events = 500
export_data_path = "data/ntuple_v3"
tree_name = "SimpleJet"
batch_size = 20
dev_size = 0.1
test_size = 0.2
featurized = False

# prop data
jet_filter = False 
jet_min_pt = 20 
jet_max_pt = 2000 
jet_min_eta = 0 
jet_max_eta = 1  
topo_filter = False 
topo_min_pt = 0 
topo_max_pt = 5 
topo_min_eta = 0 
topo_max_eta = 0.5

# features
tops = 2
feature_mode = 3
input_size = 17
output_size = 3
output_sizes = range(3, 5)
layer_extractors = dict()
for l in range(24):
    layer_extractors[l] = LayerExtractor(l, 1.5, 0.1, 1.5, 0.1)

# model
restore = False
output_path = None
dropout = 1.0
lr = 0.001
reg = 0.01
n_epochs = 20
reg_values = np.logspace(-6,0.1,20)
selection = "acc"
f1_mode = "micro"
layers = [
    Dropout(name="drop1"), 
    FullyConnected(output_size, name="fc1"),
    ]
