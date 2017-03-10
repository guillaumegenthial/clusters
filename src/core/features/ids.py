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

def get_default_processing(data, n_features, processing_y, max_length, 
        max_id, pad_tok_id, unk_tok_id, pad_tok_feat, statistics="default"):
    """
    Compute statistics over data and outputs a function
    Args:
        data: (generator) of (x, y), i tuples
        processing_y: function of a batch of Y
        max_length: padding length
        max_id: max id for cell ids
        pad_tok_id: id used for padding 
        unk_tok_id: id used for unknown cell ids (outside vocabulary)
        pad_tok_feat: feature vector used for paddings 
        statistics: string, method to use for the processing

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

    def get_tok_id(tok_id):
        if tok_id > max_id:
            return unk_tok_id
        else:
            return tok_id

    def f(X, Y):    
        ids = []
        features_batch = []
        # apply statistics
        for x in X:
            features_cluster = []
            ids += [map(get_tok_id, x["ids"])]
            for feat_cell in x["features"]:
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
                        
                features_cluster += [features_cell]
            features_batch += [features_cluster]

        ids_pad  = pad_sequences(ids, max_length, pad_tok_id)
        feat_pad = pad_sequences(features_batch, max_length, pad_tok_feat)

        return [ids_pad, feat_pad], processing_y(Y)

    return f