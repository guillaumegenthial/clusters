import importlib
from core.utils.preprocess import max_y
from core.utils.evaluate import featurized_export_result
from core.features.layers import wrap_extractor
from core.utils.general import args, apply_options
from core.dataset.pickle import make_datasets
from core.features.simple import simple_features, get_default_processing
from core.models.inputs import FlatInput

# load config
options = args("baseline")
config = importlib.import_module("configs."+options.config)
config = apply_options(config, options)

# data extraction
featurizer = simple_features(config.tops, config.feature_mode)
preprocess = lambda cluster: (featurizer(cluster), cluster["nparts"])
featurizer_raw = wrap_extractor(config.extractor)
preprocess_raw = lambda cluster: (featurizer_raw(cluster), cluster["nparts"])

# get data
train_examples, dev_set, test_set, test_raw = make_datasets(
            config, preprocess, preprocess_raw)

# data processing
processing = get_default_processing(train_examples, max_y(config.output_size))

# model
model = FlatInput(config, config.input_size)
model.build()
model.train(train_examples, dev_set, processing)
acc, base = model.evaluate(test_set, processing, test_raw, featurized_export_result)