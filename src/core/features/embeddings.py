import numpy as np
from core.utils.preprocess import pad_sequences

def ids_features(modes=["e_density"]):
    def f(d_):
        cells = d_["topo_cells"]
        ids = []
        features = []
        for id_, data_ in cells.iteritems():
            ids += [id_]
            features += [[data_[mode] for mode in modes]]

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
    eps = 0.00001
    means = np.zeros(n_features)
    var = np.zeros(n_features)
    counts = 0

    # get statistics
    for (x, y), _ in data:
        features = x["features"]
        for feat_cell in features:
            for i, feat in enumerate(feat_cell):
                means[i] += feat
                var[i] += feat**2
                counts += 1

    # compute statistics
    means /= counts
    var /= counts
    var -= means**2

    def f(X, Y):    
        ids = []
        features = []
        # apply statistics
        for x in X:
            features_ = x["features"]
            ids += [x["ids"]]
            for feat_cell in features_:
                features_cell = []
                for i, feat in enumerate(feat_cell):
                    features_cell += [(feat - means[i]) / np.sqrt(var[i] + eps)]
                    
            features += [features_cell]

        ids_pad  = pad_sequences(ids, max_length, id_tok)
        feat_pad = pad_sequences(features, max_length, feat_tok)

        if n_features == 1:
            feat_pad = np.expand_dims(feat_pad, -1)

        return [ids_pad, feat_pad], processing_y(Y)

    return f




