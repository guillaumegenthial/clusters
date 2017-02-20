from utils.features_utils import LayerExtractor

exp_name = "plots_2k"
verbose = 1 # 0 : print everything 1 : print only summary : 2 : minimal printing
max_events = 10
data_path = "data/ntuple_v3_2000k.root"
export_data_path = "data/ntuple_v3"
load_from_export_data_path = True # speedup x2000
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
results_path = "results/test_set_results.txt"
max_eta = 0.4
modes = ["e", "vol"]
layer_extractors = {
0: LayerExtractor(0, 1.5, 0.1, 1.5, 0.1),
1: LayerExtractor(1, 1.5, 0.01, 1.5, 0.1),
2: LayerExtractor(2, 1.5, 0.01, 1.5, 0.1), 
3: LayerExtractor(3, 1.5, 0.01, 1.5, 0.1),
# 4: LayerExtractor(4, 1.5, 0.025, 1.5, 0.08), 
# 5: LayerExtractor(5, 1.5, 0.025, 1.5, 0.08),
# 6: LayerExtractor(6, 1.5, 0.025, 1.5, 0.08), 
# 7: LayerExtractor(7, 1.5, 0.025, 1.5, 0.08), 
# 8: LayerExtractor(8, 1.5, 0.025, 1.5, 0.08),
# 9: LayerExtractor(9, 1.5, 0.025, 1.5, 0.08),
# 10: LayerExtractor(10, 1.5, 0.025, 1.5, 0.08),
# 11: LayerExtractor(11, 1.5, 0.025, 1.5, 0.08),
12: LayerExtractor(12, 1.5, 0.01, 1.5, 0.098),
13: LayerExtractor(13, 1.5, 0.01, 1.5, 0.098),
14: LayerExtractor(14, 1.5, 0.01, 1.5, 0.098),
# 15: LayerExtractor(15, 1.5, 0.025, 1.5, 0.08),
# 16: LayerExtractor(16, 1.5, 0.025, 1.5, 0.08),
# 17: LayerExtractor(17, 1.5, 0.025, 1.5, 0.08),
# 18: LayerExtractor(18, 1.5, 0.025, 1.5, 0.08),
# 19: LayerExtractor(19, 1.5, 0.025, 1.5, 0.08),
# 20: LayerExtractor(20, 1.5, 0.025, 1.5, 0.08),
# 21: LayerExtractor(21, 1.5, 0.025, 1.5, 0.08),
# 22: LayerExtractor(22, 1.5, 0.025, 1.5, 0.08),
# 23: LayerExtractor(23, 1.5, 0.025, 1.5, 0.08),
}