from core.utils.preprocess_utils import  default_preprocess, baseline, \
    load_and_preprocess_data, no_preprocess, one_hot
from core.utils.features_utils import simple_features, Extractor
from core.utils.general_utils import export_matrices
from core.models.regression import Regression
import config


train_examples, dev_set, test_set, test_raw = load_and_preprocess_data(config, 
    simple_features(config.tops, config.feature_mode), 
    default_preprocess, one_hot(config.output_size))

model = Regression(config, train_examples[0].shape[1])
model.train(train_examples, dev_set)
model.evaluate(test_set, test_raw)
