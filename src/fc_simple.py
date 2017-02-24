from core.utils.preprocess_utils import  default_preprocess, baseline, \
    load_and_preprocess_data, no_preprocess, one_hot, export_data
from core.utils.features_utils import simple_features, Extractor, \
    cnn_simple_features
from core.utils.general_utils import export_matrices
from core.models.regression import Regression, MultiRegression, \
    RawRegression
import config


# data extractor
extractor = Extractor(config.layer_extractors, config.modes)
n_layers = len(config.layer_extractors)
n_phi = config.layer_extractors.values()[0].n_phi
n_eta = config.layer_extractors.values()[0].n_eta
n_features = n_layers * len(config.modes)
featurizer = cnn_simple_features(extractor)
preprocess_x = default_preprocess
preprocess_y = one_hot(config.output_size)

# get data
train_examples, dev_set, test_set, test_raw = load_and_preprocess_data(config, 
                        featurizer, preprocess_x, preprocess_y)

# model
model = RawRegression(config, n_phi, n_eta, n_features)
model.build()
model.train(train_examples, dev_set)
model.evaluate(test_set, test_raw)
