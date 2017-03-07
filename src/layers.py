import importlib
from core.utils.preprocess import default_preprocess, \
    no_preprocess, one_hot, max_y 
from core.utils.data import load_and_preprocess_data, export_data
from core.features.layers import Extractor, wrap_extractor, \
    extractor_default_preprocess, extractor_post_process
from core.utils.general import args, apply_options
from core.models.inputs import SquareInput
from core.utils.evaluate import featurized_export_result


# load config
options = args("fc")
config = importlib.import_module("configs."+options.config)
config = apply_options(config, options)

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
acc, base = model.evaluate(test_set, post_process, [test_raw, test_set],
                         featurized_export_result)

