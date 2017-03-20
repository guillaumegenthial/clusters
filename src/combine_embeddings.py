import importlib
from core.utils.preprocess import preprocess_y
from core.utils.evaluate import featurized_export_result, export_clustering
from core.features.layers import wrap_extractor
from core.features.embeddings import embedding_features, get_default_processing
from core.utils.general import args, apply_options
from core.dataset.pickle import make_datasets
from core.models.input import EmbeddingsInput
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool, Combine, Reduce, Embedding

# load config
options = args("embeddings")
config = importlib.import_module("configs."+options.config)
config = apply_options(config, options)

# data featurizer
featurizer = embedding_features(config.modes)
preprocess = lambda cluster: (featurizer(cluster), cluster["nparts"], cluster["props"])
featurizer_raw = wrap_extractor(config.extractor)
preprocess_raw = lambda cluster: (featurizer_raw(cluster), cluster["nparts"])

# get data
train_examples, dev_set, test_set, test_raw = make_datasets(
            config, preprocess, preprocess_raw)

# data processing
processing = get_default_processing(train_examples, config.n_features, 
    preprocess_y(1, 3), config.max_n_cells, config.pad_tok, 
    config.preprocessing_mode)

# model
model = EmbeddingsInput(config)
model.build("light")

path1 = "results/20170318_165602/model.weights/"
path2 = "results/20170318_234736/model.weights/"

model.combine(path1, path2, processing, test_set)