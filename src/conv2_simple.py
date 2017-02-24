from core.utils.preprocess_utils import  default_preprocess, baseline, \
    load_and_preprocess_data, no_preprocess, one_hot, export_data, \
    max_y, load_data_featurized
from core.utils.features_utils import simple_features, Extractor, \
    wrap_extractor, extractor_default_preprocess, extractor_post_process
from core.utils.general_utils import export_matrices
from core.models.regression import Regression, MultiRegression, \
    RawRegression
from core.models.cnn import Conv2d
import config


# data extractor
extractor = Extractor(config.layer_extractors, config.modes)
n_layers = len(config.layer_extractors)
n_phi = config.layer_extractors.values()[0].n_phi
n_eta = config.layer_extractors.values()[0].n_eta
n_features = n_layers * len(config.modes)
featurizer = wrap_extractor(extractor)
preprocess = extractor_default_preprocess(extractor)
post_process = extractor_post_process(extractor, config.output_size)

# get data
train_examples, dev_set, test_set, test_raw = load_and_preprocess_data(config, 
                                                featurizer, preprocess)

# model
model = Conv2d(config, n_phi, n_eta, n_features)
model.build()
model.train(train_examples, dev_set, post_process)
model.evaluate(test_set, test_raw, post_process)
