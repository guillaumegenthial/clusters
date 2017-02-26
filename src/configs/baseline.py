import numpy as np
from core.utils.features_utils import LayerExtractor
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool

# general
exp_name = "plots_2k"

# data
data_path = "data/ntuple_v3_2000k.root"
data_verbosity = 2
max_events = 10
export_data_path = "data/ntuple_v3"
tree_name = "SimpleJet"
batch_size = 20
dev_size = 0.1
test_size = 0.2
max_eta = 0.5
min_energy = 20
featurized = False

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
output_path = None
dropout = 1.0
lr = 0.001
reg = 0.01
n_epochs = 1
reg_values = np.logspace(-6,0.1,20)

layers = [
    Dropout(name="drop1"), 
    FullyConnected(10, name="fc1"),
    ReLu(),
    Dropout(name="drop2"), 
    FullyConnected(output_size, name="fc2")
    ]