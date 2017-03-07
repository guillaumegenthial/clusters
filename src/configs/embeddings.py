import numpy as np
from core.utils.features_utils import LayerExtractor
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool

# general
exp_name = "embeddings"

# data
data_path = "data/ntuple_v3_2000k.root"
data_verbosity = 2
max_events = 100
export_data_path = "data/ntuple_v3"
tree_name = "SimpleJet"
batch_size = 20
dev_size = 0.1
test_size = 0.2
max_eta = 0.5
min_energy = 20
featurized = False

# features
n_cells = 30000
max_n_cells = 10
modes = ["e"]
n_features = len(modes)
embedding_size = 50
id_tok = 0
feat_tok = np.zeros(len(modes))
output_size = 3
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
