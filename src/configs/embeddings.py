import numpy as np
from core.features.layers import LayerExtractor, Extractor
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool, Combine, Reduce, Embedding, Concat, Expand, \
    LastConcat, Mask, Squeeze
import tensorflow as tf

# general
exp_name = "embeddings with dropout and without l2"

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
max_n_cells = 300
modes = ["e", "e_density", "eta", "phi", "vol", "pT", "dep"]
n_features = 6 + 24
embedding_size = 64
pad_tok = np.zeros(n_features)
output_size = 2
layer_extractors = dict()
for l in range(24):
    layer_extractors[l] = LayerExtractor(l, 1.5, 0.1, 1.5, 0.1)
extractor = Extractor(layer_extractors, ["e", "vol", "e_density"])

# model
baseclass = 0
batch_size = 20
n_epochs = 15
restore = False
output_path = None
dropout = 0.8
lr = 0.001
reg = 0.01
reg_values = np.logspace(-6,0.1,20)
selection = "acc"
f1_mode = "micro"
layers = [
    # embedding
    FullyConnected(embedding_size, name="embedding"),
    FullyConnected(2*embedding_size, name="embed_fc1"), 
    ReLu(name="embed_relu1"), 
    FullyConnected(16*embedding_size, name="embed_fc2"), 
    ReLu(name="embed_relu2"),
    Mask(val=0, name="mask"),
    Reduce(axis=1, op="max", keep_dims=True, name="max_pool"), 
    Squeeze(name="global"),
    LastConcat(axis=2, input_names=["embedding", "max_pool"], name="concat"), 
    FullyConnected(12*embedding_size, name="max_fc1"), 
    ReLu(name="max_relu1"), 
    FullyConnected(8*embedding_size, name="max_fc2"), 
    ReLu(name="max_relu2"),
    Mask(val=0, name="mask"),
    Reduce(axis=1, op="max", keep_dims=False, name="max_pool_readout"),
    Concat(axis=1, name="concat2", input_names=["max_pool_readout", "global"]), 
    FullyConnected(8*embedding_size, name="flatten_fc1"), 
    ReLu(name="flatten_relu1"),
    FullyConnected(4*embedding_size, name="flatten_fc2"), 
    ReLu(name="flatten_relu2"),
    FullyConnected(2*embedding_size, name="flatten_fc3"), 
    ReLu(name="flatten_relu3"),
    Dropout(name="final_drop"),
    FullyConnected(output_size, name="output_layer")
]