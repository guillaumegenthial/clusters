import importlib
from core.utils.preprocess import preprocess_y
from core.utils.evaluate import featurized_export_result
from core.features.layers import wrap_extractor
from core.features.embeddings import ids_features, get_default_processing
from core.utils.general import args, apply_options
from core.dataset.pickle import make_datasets
from core.models.inputs import EmbeddingsInput

# load config
options = args("embeddings")
config = importlib.import_module("configs."+options.config)
config = apply_options(config, options)

# data featurizer
featurizer = ids_features(config.modes)
preprocess = lambda cluster: (featurizer(cluster), cluster["nparts"])
featurizer_raw = wrap_extractor(config.extractor)
preprocess_raw = lambda cluster: (featurizer_raw(cluster), cluster["nparts"])

# get data
train_examples, dev_set, test_set, test_raw = make_datasets(
            config, preprocess, preprocess_raw)

# data processing
processing = get_default_processing(train_examples, config.n_features, 
    preprocess_y(config.output_size), config.max_n_cells, config.id_tok, config.feat_tok)

# model
model = EmbeddingsInput(config)
model.build()
model.train(train_examples, dev_set, processing)
acc, base = model.evaluate(test_set, processing, test_raw, featurized_export_result)
