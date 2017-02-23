import numpy as np
from core.utils.preprocess_utils import  default_preprocess, baseline, \
    load_and_preprocess_data, no_preprocess, one_hot, export_data, \
    preprocess_data, load_data_raw_it, extract_data_it, it_to_list
from core.utils.features_utils import simple_features, Extractor
from core.utils.general_utils import export_matrices, simplePlot
from core.models.regression import Regression, MultiRegression
import config

featurizer, _ = simple_features(config.tops, config.feature_mode)
preprocess_x = default_preprocess
preprocess_y = one_hot(config.output_size)
train_examples, dev_set, test_set, test_raw = load_and_preprocess_data(
                            config, featurizer, preprocess_x, preprocess_y, True)

# train_examples, dev_set, test_set, test_raw = load_and_preprocess_data(
#                             config, featurizer, preprocess_x, preprocess_y, False)

input_size = train_examples[0][0].shape[0]
model = MultiRegression(config, input_size)
model.build()
model.train(train_examples, dev_set)
acc, base = model.evaluate(test_set)



# train_, dev_, test_, test_raw = load_and_preprocess_data(config, features)
# accuracies = []
# for output_size in config.output_sizes:
#     config.output_size = output_size
#     preprocess_x = default_preprocess
#     preprocess_y = one_hot(config.output_size)

#     train_examples = preprocess_data(train_, preprocess_x, preprocess_y)
#     dev_set = preprocess_data(dev_, preprocess_x, preprocess_y)
#     test_set = preprocess_data(test_, preprocess_x, preprocess_y)

#     input_size = train_examples[0].shape[1]
#     model = MultiRegression(config, input_size)
#     model.build()

#     model.train(train_examples, dev_set)
#     acc, base = model.evaluate(test_set)
#     accuracies.append((acc, base))


# filename = model.config.output_path + "output_size_vs_acccuracy.png"
# simplePlot(config.output_sizes, accuracies, filename, "output size", "accuracy", )
