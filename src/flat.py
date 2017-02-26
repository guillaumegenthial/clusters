import numpy as np
import importlib
from core.utils.preprocess_utils import  default_preprocess, \
    load_and_preprocess_data, no_preprocess, one_hot, export_data, \
    make_preprocess, load_data_raw_it, extract_data_it, it_to_list, \
    max_y
from core.utils.features_utils import simple_features, Extractor
from core.utils.general_utils import args
from core.models.inputs import FlatInput
from core.models.layer import FullyConnected, Dropout, Flatten, \
    ReLu, Conv2d, MaxPool
from core.utils.evaluate_utils import raw_export_result

# load config
options = args()
config = importlib.import_module("configs."+options.config)

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