import importlib
from core.utils.preprocess import preprocess_y
from core.utils.evaluate import featurized_export_result
from core.features.layers import wrap_extractor
from core.features.ids import ids_features, get_default_processing
from core.utils.general import args, apply_options
from core.dataset.pickle import make_datasets
from core.models.input import IdInput

# load config
options = args("ids")
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
all_ids = set()
for (x, y), i in train_examples:
    all_ids.update(x["ids"])

# dynamically allocate the vocab size 
config.n_cells = max(all_ids) + 3
config.unk_tok_id = config.n_cells - 1
config.pad_tok_id = config.n_cells - 2
for layer in config.layers:
    if layer.__class__.__name__ == "Embedding":
        print "Setting vocab_size to {}".format(config.n_cells)
        layer.vocab_size = config.n_cells

# data processing
processing = get_default_processing(train_examples, config.n_features, 
    preprocess_y(config.output_size), config.max_n_cells, config.n_cells-3, 
    config.pad_tok_id, config.unk_tok_id, config.pad_tok_feat, "none")

# model
model = IdInput(config)
model.build()
model.train(train_examples, dev_set, processing)
acc, base = model.evaluate(test_set, processing, test_raw, featurized_export_result)
