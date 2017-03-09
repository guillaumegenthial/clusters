import numpy as np
from core.features.layers import LayerExtractor, Extractor
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool

# general
exp_name = "plots_2k"

# general data
path = "data/events"
max_iter = 200
train_files = "data/config_{}/train.txt".format(max_iter)
dev_files = "data/config_{}/dev.txt".format(max_iter)
test_files = "data/config_{}/test.txt".format(max_iter)
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
tops = 2
feature_mode = 3
input_size = 17
output_size = 3
output_sizes = range(3, 5)
layer_extractors = dict()
for l in range(24):
    layer_extractors[l] = LayerExtractor(l, 1.5, 0.1, 1.5, 0.1)

modes = ["e", "vol"]
extractor = Extractor(layer_extractors, modes)
n_layers = len(layer_extractors)
n_phi = layer_extractors.values()[0].n_phi
n_eta = layer_extractors.values()[0].n_eta
n_features = n_layers * len(modes)

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
    Conv2d(3, 3, n_features, 100, name="conv1"),
    ReLu(),
    MaxPool(name="pool1"),
    Flatten(),
    FullyConnected(1000, name="fc1"),
    ReLu(),
    Dropout(),
    FullyConnected(output_size, name="fc2"),
    ]

