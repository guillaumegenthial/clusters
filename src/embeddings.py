import numpy as np
import importlib
from core.utils.preprocess_utils import  default_preprocess, \
    load_and_preprocess_data, no_preprocess, one_hot, export_data, \
    make_preprocess, load_data_raw_it, extract_data_it, it_to_list, \
    max_y, pad_post_process
from core.utils.features_utils import ids_features, ids_preprocess
from core.utils.general_utils import args
from core.models.inputs import EmbeddingsInput
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool
from core.utils.evaluate_utils import raw_export_result

# load config
options = args()
config = importlib.import_module("configs."+options.config)
if options.test:
    config.max_events = 10
    config.n_epochs = 2

if options.restore:
    config.restore = True

if options.epochs != 20:
    config.n_epochs = options.epochs

# data featurizer
featurizer = ids_features(config.modes)
preprocess = ids_preprocess(config.n_features)

# get data
train_examples, dev_set, test_set, test_raw = load_and_preprocess_data(
                            config, featurizer)

# model
post_process = pad_post_process(config.max_n_cells, 
    config.id_tok, config.feat_tok, config.output_size)
model = EmbeddingsInput(config)
model.build()
model.train(train_examples, dev_set, post_process)
acc, base = model.evaluate(test_set, post_process, test_raw=test_raw, 
    export_result=raw_export_result)