import importlib
import numpy as np
from core.utils.preprocess import  default_preprocess, max_y, \
    no_preprocess, one_hot, make_preprocess
from core.utils.data import load_and_preprocess_data, \
    load_data_raw_it, extract_data_it, it_to_list, export_data
from core.features.simple import simple_features
from core.utils.general import args, apply_options
from core.models.inputs import FlatInput
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool
from core.utils.evaluate import raw_export_result

# load config
options = args("baseline")
config = importlib.import_module("configs."+options.config)
config = apply_options(config, options)

# data featurizer
featurizer, _ = simple_features(config.tops, config.feature_mode)
preprocess_x = default_preprocess
preprocess_y = max_y(config.output_size)
preprocess = make_preprocess(preprocess_x, preprocess_y)

# get data
train_examples, dev_set, test_set, test_raw = load_and_preprocess_data(
                            config, featurizer, preprocess)


# model
model = FlatInput(config, config.input_size)
model.build()
model.train(train_examples, dev_set)
acc, base = model.evaluate(test_set, test_raw=test_raw, export_result=raw_export_result)