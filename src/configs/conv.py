import numpy as np
from core.features.layers import LayerExtractor, Extractor
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool
from general import general


class config(general):
    # general
    exp_name = "conv"

    # features
    tops = 2
    feature_mode = 3
    input_size = 17
    layer_extractors = dict()
    for l in range(24):
        layer_extractors[l] = LayerExtractor(l, 1.5, 0.1, 1.5, 0.1)

    modes = ["e", "vol", "e_density"]
    extractor = Extractor(layer_extractors, modes)
    n_layers = len(layer_extractors)
    n_phi = layer_extractors.values()[0].n_phi
    n_eta = layer_extractors.values()[0].n_eta
    n_features = n_layers * len(modes)

    # model
    output_path = None
    layers = [
        Conv2d(3, 3, n_features, 100, name="conv1"),
        ReLu(name="conv_relu1"),
        MaxPool(name="conv_pool1"),
        Flatten(name="flatten"),
        FullyConnected(1000, name="fc1"),
        ReLu(name="relu1"),
        Dropout(name="drop"),
        FullyConnected(output_size, name="output_layer"),
        ]

