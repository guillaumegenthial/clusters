from utils.preprocess_utils import  default_preprocess, baseline, \
                load_and_preprocess_data, no_preprocess, one_hot
from utils.features_utils import simple_features, get_simple_features
import config
from regression import Regression


features = get_simple_features(config.tops, config.feature_mode)
train_examples, dev_set, test_set = load_and_preprocess_data(config, 
                        features, default_preprocess, one_hot)
dev_baseline  = baseline(dev_set)
test_baseline = baseline(test_set)

input_size = train_examples[0].shape[1]

model = Regression(config, input_size)
model.train(train_examples, dev_set, dev_baseline, test_set, test_baseline)

