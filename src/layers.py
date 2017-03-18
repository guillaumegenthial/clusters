import importlib
from core.utils.general import args, apply_options
from core.utils.evaluate import featurized_export_result
from core.utils.preprocess import preprocess_y
from core.dataset.pickle import make_datasets
from core.features.layers import wrap_extractor, get_default_processing
from core.models.input import SquareInput


# load config
options = args("fc")
config = importlib.import_module("configs."+options.config)
config = apply_options(config, options)

# data featurizers
extractor = config.extractor
featurizer = wrap_extractor(extractor)
preprocess = lambda cluster: (featurizer(cluster), cluster["nparts"], cluster["props"])

# get data
train_examples, dev_set, test_set, test_raw = make_datasets(config, preprocess, preprocess)

# data processing
processing = get_default_processing(train_examples, extractor, preprocess_y(config.part_min, config.output_size))

# model
model = SquareInput(config, config.n_eta, config.n_phi, config.n_features)
model.build()
model.train(train_examples, dev_set, processing)
acc, base = model.evaluate(test_set, processing, test_raw, featurized_export_result)

