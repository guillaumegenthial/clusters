import numpy as np
from core.utils.preprocess_utils import  default_preprocess, baseline, \
    load_and_preprocess_data, no_preprocess, one_hot, export_data
from core.utils.features_utils import simple_features, Extractor
from core.utils.general_utils import export_matrices
from core.models.regression import Regression, MultiRegression
import config

reg_values = np.logspace(-6,0.1,20)
features, _ = simple_features(config.tops, config.feature_mode)

train_examples, dev_set, test_set, test_raw = load_and_preprocess_data(config, 
                    features, default_preprocess, one_hot(config.output_size))
# export_data(train_examples, modes, "data/features.txt")


input_size = train_examples[0].shape[1]

model = MultiRegression(config, input_size)
model.build()

if config.train_mode == "train":
    model.train(train_examples, dev_set)
    model.evaluate(test_set, test_raw)

elif config.train_mode == "reg":
    model.find_best_reg_value(reg_values, train_examples, dev_set, test_set)