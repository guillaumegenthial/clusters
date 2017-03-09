import numpy as np
from core.features.layers import LayerExtractor, Extractor
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool

# general
exp_name = "baseline"

# general data
path = "data/events"
train_files = "data/config/train.txt"
dev_files = "data/config/dev.txt"
test_files = "data/config/test.txt"
max_iter = 50
shuffle = True
dev_size = 0.1
test_size = 0.2

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
tops = 5
feature_mode = 3
input_size = 24
output_size = 3
layer_extractors = dict()
for l in range(24):
    layer_extractors[l] = LayerExtractor(l, 1.5, 0.1, 1.5, 0.1)

modes = ["e", "vol"]
extractor = Extractor(layer_extractors, modes)

# model
baseclass = 0
batch_size = 20
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
    FullyConnected(20, name="fc1"),
    ReLu(),
    Dropout(),
    FullyConnected(20, name="fc2"),
    ReLu(),
    Dropout(),
    FullyConnected(output_size, name="fc3")
    ]
