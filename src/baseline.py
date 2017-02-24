import numpy as np
from core.utils.preprocess_utils import  default_preprocess, baseline, \
    load_and_preprocess_data, no_preprocess, one_hot, export_data, \
    make_preprocess, load_data_raw_it, extract_data_it, it_to_list, \
    max_y
from core.utils.features_utils import simple_features, Extractor
from core.utils.general_utils import export_matrices, simplePlot
from core.models.regression import Regression, MultiRegression
import config

featurizer, _ = simple_features(config.tops, config.feature_mode)

if config.exp_mode == "test":
    preprocess_x = default_preprocess
    preprocess_y = max_y(config.output_size)
    preprocess = make_preprocess(preprocess_x, preprocess_y)
    train_examples, dev_set, test_set, test_raw = load_and_preprocess_data(
                                config, featurizer, preprocess)

    input_size = train_examples[0][0].shape[0]
    model = MultiRegression(config, input_size)
    model.build()
    model.train(train_examples, dev_set)
    acc, base = model.evaluate(test_set)

elif config.exp_mode == "evolution":
    train_, dev_, test_, test_raw = load_and_preprocess_data(config, featurizer)
    accuracies = []
    for output_size in config.output_sizes:
        config.output_size = output_size
        preprocess_x = default_preprocess
        preprocess_y = max_y(config.output_size)

        preprocess = make_preprocess(preprocess_x, preprocess_y)

        train_examples = preprocess(train_)
        dev_set = preprocess(dev_)
        test_set = preprocess(test_)

        input_size = train_examples[0][0].shape[0]
        model = MultiRegression(config, input_size)
        model.build()

        model.train(train_examples, dev_set)
        acc, base = model.evaluate(test_set)
        accuracies.append((acc, base))


    filename = model.config.output_path + "output_size_vs_acccuracy.png"
    simplePlot(config.output_sizes, accuracies, filename, "output size", "accuracy", )
