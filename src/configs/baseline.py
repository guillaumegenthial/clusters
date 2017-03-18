import numpy as np
from core.features.layers import LayerExtractor, Extractor
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool
from general import general

class config(general):
    exp_name = "baseline"

    # features
    tops = 5
    feature_mode = 3
    input_size = 24
    layer_extractors = dict()
    for l in range(24):
        layer_extractors[l] = LayerExtractor(l, 1.5, 0.1, 1.5, 0.1)

    modes = ["e", "vol", "e_density"]
    extractor = Extractor(layer_extractors, modes)

    # model
    output_path = None
    layers = [
        FullyConnected(general.output_size, name="fc1"),
        ]


