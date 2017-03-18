import numpy as np
from core.features.layers import LayerExtractor, Extractor
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool, Combine, Reduce, Embedding, Concat, Expand, \
    LastConcat, Mask, Squeeze
import tensorflow as tf
from general import general

class config(general):
    # general
    exp_name = "embeddings"

    # features
    max_n_cells = 150
    modes = ["e", "e_density", "eta", "phi", "vol", "pT", "dep"]
    n_features = 7
    embedding_size = 64
    pad_tok = np.zeros(n_features)
    output_size = 2
    layer_extractors = dict()
    for l in range(24):
        layer_extractors[l] = LayerExtractor(l, 1.5, 0.1, 1.5, 0.1)
    extractor = Extractor(layer_extractors, ["e", "vol", "e_density"])

    # model
    output_path = None
    layers = [
        # embedding
        FullyConnected(embedding_size, name="embedding"),
        FullyConnected(2*embedding_size, name="embed_fc1"), 
        # ReLu(name="embed_relu1"), 
        # FullyConnected(16*embedding_size, name="embed_fc2"), 
        ReLu(name="embed_relu2"),
        Mask(val=-10000, name="mask"),
        Reduce(axis=1, op="max", keep_dims=True, name="max_pool"), 
        Squeeze(name="global", axis=1),
        LastConcat(axis=2, input_names=["embedding", "max_pool"], name="concat"), 
        # FullyConnected(12*embedding_size, name="max_fc1"), 
        # ReLu(name="max_relu1"), 
        FullyConnected(2*embedding_size, name="max_fc2"), 
        ReLu(name="max_relu2"),
        Mask(val=-10000, name="mask"),
        Reduce(axis=1, op="max", keep_dims=False, name="max_pool_readout"),
        Concat(axis=1, name="concat2", input_names=["max_pool_readout", "global"]), 
        # FullyConnected(8*embedding_size, name="flatten_fc1"), 
        # ReLu(name="flatten_relu1"),
        # FullyConnected(4*embedding_size, name="flatten_fc2"), 
        # ReLu(name="flatten_relu2"),
        FullyConnected(2*embedding_size, name="flatten_fc3"), 
        ReLu(name="flatten_relu3"),
        # Dropout(name="final_drop"),
        FullyConnected(general.output_size, name="output_layer")
    ]