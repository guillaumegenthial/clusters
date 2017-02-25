import importlib
from core.utils.preprocess_utils import  default_preprocess, baseline, \
    load_and_preprocess_data, no_preprocess, one_hot, export_data, \
    max_y, load_data_featurized
from core.utils.features_utils import simple_features, Extractor, \
    wrap_extractor, extractor_default_preprocess, extractor_post_process
from core.utils.general_utils import export_matrices, args
from core.models.inputs import SquareInput

# import config
options = args()
config = importlib.import_module("configs."+options.config)

# data featurizers
extractor = config.extractor
featurizer = wrap_extractor(extractor)
preprocess = extractor_default_preprocess(extractor)
post_process = extractor_post_process(extractor, config.output_size)

# get data
train_examples, dev_set, test_set, test_raw = load_and_preprocess_data(config, 
                                                featurizer, preprocess)

# model
model = SquareInput(config, config.n_eta, config.n_phi, config.n_features)
model.build()
model.train(train_examples, dev_set, post_process)
model.evaluate(test_set, test_raw, post_process)

