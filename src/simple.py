import importlib
import numpy as np
from core.dataset.pickle import make_datasets
from core.utils.preprocess import  default_preprocess, max_y, \
    no_preprocess, one_hot, make_preprocess
from core.utils.data import load_and_preprocess_data, \
    load_data_raw_it, extract_data_it, it_to_list, export_data
from core.features.simple import simple_features, get_default_processing
from core.utils.general import args, apply_options
from core.models.inputs import FlatInput
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool
from core.utils.evaluate import raw_export_result

# load config
options = args("baseline")
config = importlib.import_module("configs."+options.config)
config = apply_options(config, options)

# data extraction
featurizer = simple_features(config.tops, config.feature_mode)
preprocess = lambda cluster: (featurizer(cluster), cluster["nparts"])

# get data
train_examples, dev_set, test_set = make_datasets(config, preprocess)

# data processing
processing = get_default_processing(train_examples, config.output_size)

# model
model = FlatInput(config, config.input_size)
model.build()
model.train(train_examples, dev_set, processing)
acc, base = model.evaluate(test_set, processing)