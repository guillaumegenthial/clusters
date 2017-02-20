from utils.features_utils import LayerExtractor

exp_name = "plots_2k"
verbose = 1 # 0 : print everything 1 : print only summary : 2 : minimal printing
max_events = 10
data_path = "data/ntuple_v3_2000k.root"
export_data_path = "data/ntuple_v3"
load_from_export_data_path = True # speedup x2000
output_path = "results/"
tree_name = "SimpleJet"
max_eta = 0.5
min_energy = 20
tops = 5
feature_mode = 1
input_size = 7 + tops
output_size = 3
hidden_size = output_size
dropout = 0.5
lr = 0.001
n_epochs = 10
batch_size = 20
data_verbosity = 2
dev_size = 0.1
test_size = 0.2
max_eta = 0.4
modes = ["e", "vol"]
layer_extractors = dict()
for l in range(24):
    layer_extractors[l] = LayerExtractor(l, 1.5, 0.1, 1.5, 0.1)