from core.utils.general import Progbar
import numpy as np
from utils import get_mode
from core.utils.preprocess import pad_sequences


def embedding_features(modes):
    """
    Featurizer
    Args:
        modes: list of string
    Return:
        a function that takes a cluster as input
        f(d_) = {ids: ..., features: ...}
    """
    def f(d_):
        cells = d_["topo_cells"]
        features = []
        for id_, cell_ in cells.iteritems():
            res = []
            for mode in modes:
                feat = get_mode(cell_, mode) 
                if type(feat) == list:
                    res += feat
                else:
                    res += [feat]

            features += [res]

        return features

    return f


def get_default_processing(data, n_features, processing_y, max_length, pad_tok, statistics="default"):
    """
    Compute statistics over data and outputs a function
    Args:
        data: (generator) of (x, y), i tuples
        processing_y: function of a batch of Y
        max_length: padding length
        pad_tok: used for padding
    Return:
        function f(x, y) where x, y are batches of x and y
    """

    print "Creating processing function"
    eps = 0.00001
    means = np.zeros(n_features)
    var = np.zeros(n_features)
    counts = 0

    prog = Progbar(target=data.max_iter)
    n_examples = 0
    # get statistics
    for (x, y), n_event in data:
        n_examples += 1
        prog.update(n_event)
        for feat_cell in x:
            for i, feat in enumerate(feat_cell):
                means[i] += feat
                var[i] += feat**2
                counts += 1

    data.length = n_examples
    prog.update(n_event+1)

    # compute statistics
    means /= counts
    var /= counts
    var -= means**2

    print "- done."


    def f(X, Y):
        features_batch = []
        # apply statistics
        for x in X:
            features_cluster = []
            for feat_cell in x:
                features_cell = []
                for i, feat in enumerate(feat_cell):
                    if statistics == "default":
                        features_cell += [(feat - means[i]) / np.sqrt(var[i] + eps)]
                    elif statistics == "mean":
                        features_cell += [(feat - means[i])]  
                    elif statistics == "scale":
                        features_cell += [feat / np.sqrt(var[i] + eps)]  
                    elif statistics == "none":
                        features_cell += [feat] 
                    elif statistics == "custom":
                        if i < 6:
                            features_cell += [(feat - means[i]) / np.sqrt(var[i] + eps)]
                        else:
                            features_cell += [feat]
                            
                features_cluster += [features_cell]
            features_batch += [features_cluster] 

        X, mask = pad_sequences(features_batch, max_length, pad_tok)
        return X, processing_y(Y), mask

    return f