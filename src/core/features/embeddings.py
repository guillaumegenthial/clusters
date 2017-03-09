from core.utils.general import Progbar
import numpy as np
from utils import get_mode
from core.utils.preprocess import pad_sequences

def ids_features(modes=["e_density"]):
    """
    Featurizer
    Args:
        modes: list of string
    Returns:
        a function that takes a cluster as input
        f(d_) = {ids: ..., features: ...}
    """
    def f(d_):
        cells = d_["topo_cells"]
        ids = []
        features = []
        for id_, cell_ in cells.iteritems():
            ids += [id_]
            res_ = []
            for mode in modes:
                res_ += [get_mode(cell_, mode)]
            features += [res_]

        return {"ids": ids, "features": features}

    return f

def get_default_processing(data, n_features, processing_y, max_length, id_tok, feat_tok):
    """
    Compute statistics over data and outputs a function
    Args:
        data: (generator) of (x, y), i tuples
    Returns:
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
        features = x["features"]
        for feat_cell in features:
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
        ids = []
        features_batch = []
        # apply statistics
        for x in X:
            features_cluster = []
            ids += [x["ids"]]
            for feat_cell in x["features"]:
                features_cell = []
                for i, feat in enumerate(feat_cell):
                    features_cell += [(feat - means[i]) / np.sqrt(var[i] + eps)]  
                features_cluster += [features_cell]
            features_batch += [features_cluster]

        ids_pad  = pad_sequences(ids, max_length, id_tok)
        feat_pad = pad_sequences(features_batch, max_length, feat_tok)

        if n_features == 1:
            feat_pad = np.expand_dims(feat_pad, -1)

        return [ids_pad, feat_pad], processing_y(Y)

    return f