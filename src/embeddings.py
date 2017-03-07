import importlib
import numpy as np
from core.utils.preprocess import default_preprocess, \
    no_preprocess, one_hot, max_y 
from core.utils.data import load_and_preprocess_data, export_data
from core.features.embeddings import ids_features, ids_preprocess, \
    ids_post_process
from core.utils.general import args, apply_options
from core.models.inputs import EmbeddingsInput
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool
from core.utils.evaluate import raw_export_result

# load config
options = args("embeddings")
config = importlib.import_module("configs."+options.config)
config = apply_options(config, options)

# data featurizer
featurizer = ids_features(config.modes)
preprocess = ids_preprocess(config.n_features)
post_process = ids_post_process(config.max_n_cells, 
    config.id_tok, config.feat_tok, config.output_size)

# get data
train_examples, dev_set, test_set, test_raw = load_and_preprocess_data(
                            config, featurizer, preprocess)

# model
model = EmbeddingsInput(config)
model.build()
model.train(train_examples, dev_set, post_process)
acc, base = model.evaluate(test_set, post_process, test_raw=test_raw, 
    export_result=raw_export_result)