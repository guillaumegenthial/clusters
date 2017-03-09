import numpy as np
from core.features.layers import LayerExtractor, Extractor
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool, Combine, Reduce, Embedding, Concat

# general
exp_name = "embeddings"

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
n_cells = None # to be set depending on the train set
max_n_cells = 200
modes = ["e", "vol", "e_density"]
n_features = len(modes)
embedding_size = 64
unk_tok_id = None
pad_tok_id = None
pad_tok_feat = np.zeros(len(modes))
output_size = 2
layer_extractors = dict()
for l in range(24):
    layer_extractors[l] = LayerExtractor(l, 1.5, 0.1, 1.5, 0.1)
extractor = Extractor(layer_extractors, ["e", "vol", "e_density"])

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
    Embedding(n_cells, embedding_size, name="embedding", input_names=["ids"]), 
    Combine(name="combine", input_names=["embedding", "features"]), 
    Reduce(axis=1, op="max", name="reduce_max", input_names=["combine"]), 
    Reduce(axis=1, op="mean", name="reduce_mean", input_names=["combine"]),
    Concat(axis=1, input_names=["reduce_max", "reduce_mean"]), 
    Flatten(),
    # FullyConnected(50), 
    # ReLu(),
    FullyConnected(output_size)
]