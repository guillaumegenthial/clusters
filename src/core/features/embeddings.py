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

def ids_preprocess(n_features):
    def f(data):
        eps = 0.00001
        means = np.zeros(n_features)
        var = np.zeros(n_features)
        counts = 0

        # get statistics
        for x, y in data:
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

        # apply statistics
        for i in range(len(data)):
            for j in range(len(data[i][0]["features"])):
                for k, feat in enumerate(data[i][0]["features"][j]):
                    data[i][0]["features"][j][k] = (feat - means[k]) / np.sqrt(var[k] + eps)

        return data

    return f

def ids_post_process(max_length, id_tok, feat_tok, output_size):
    def f(X, Y):
        ids      = [x_["ids"] for x_ in X]
        features = [x_["features"] for x_ in X]

        ids_pad  = pad_sequences(ids, max_length, id_tok)
        feat_pad = pad_sequences(features, max_length, feat_tok)

        return [ids_pad, feat_pad], np.minimum(Y, output_size-1)

    return f




