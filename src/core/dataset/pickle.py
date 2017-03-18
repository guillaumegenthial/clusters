import random
from base import DatasetBase
from core.utils.general import pickle_load, get_all_files, check_file, \
    check_dir


def make_datasets(config, preprocess=None, preprocess_raw=None):
    """
    Given a path to data files, returns 3 instances of DatasetPickle
    If config files (txt files with file names) exist uses these names
        else, does the splitting and write the config files
    Args:
        config (module or instance) with attributes used in the function
        preprocess (function) takes cluster and outputs (x, y)
    Returns:
        train, dev, test, test_raw generators
    """
    train_config_file = config.train_files
    dev_config_file   = config.dev_files
    test_config_file  = config.test_files

    if check_file(train_config_file) and check_file(dev_config_file) and check_file(test_config_file):
        # load list of file names from txt files
        files_train = load_files_from_config(train_config_file)
        files_dev   = load_files_from_config(dev_config_file)
        files_test  = load_files_from_config(test_config_file)
    else:
        # get all files in path
        # force the shuffling
        files       = get_all_files(config.path, True)[:config.max_iter]
        index_dev   = int(config.dev_size*len(files))
        index_test  = int(config.test_size*len(files))
        # list of file names
        files_dev   = files[:index_dev]
        files_test  = files[index_dev: index_dev + index_test]
        files_train = files[index_dev + index_test:]
        # dum this list for the future
        dump_files_to_config(files_train, train_config_file)
        dump_files_to_config(files_dev, dev_config_file)
        dump_files_to_config(files_test, test_config_file)

    train = DatasetPickle(config=config, files=files_train, preprocess=preprocess, 
        max_iter=int((1-config.dev_size-config.test_size)*config.max_iter))

    dev = DatasetPickle(config=config, files=files_dev, preprocess=preprocess, 
        max_iter=int(config.dev_size*config.max_iter))

    test = DatasetPickle(config=config, files=files_test, preprocess=preprocess, 
        max_iter=int(config.test_size*config.max_iter))

    test_raw = DatasetPickle(config=config, files=files_test, preprocess=preprocess_raw, 
        max_iter=int(config.test_size*config.max_iter))

    return train, dev, test, test_raw

def load_files_from_config(path):
    l = []
    with open(path, "r") as f:
        for line in f:
            l += [line.strip()]

    return l

def dump_files_to_config(l, path):
    check_dir("/".join(path.split("/")[:-1]))
    with open(path, "w") as f:
        for e in l:
            f.write(e+"\n")

class DatasetPickle(DatasetBase):
    def __init__(self, config, files, max_iter, preprocess=None):
        # base init
        DatasetBase.__init__(self)

        # not from config
        if files is None:
            self.files = get_all_files(self.path, True)
        else:
            self.files = files
        self.preprocess = preprocess
        self.max_iter = max_iter
  
        # general
        self.path = config.path
        self.shuffle = config.shuffle
        self.shuffle_clusters = config.shuffle_clusters

        # filter on jets
        self.jet_filter = config.jet_filter
        self.jet_min_pt = config.jet_min_pt
        self.jet_max_pt = config.jet_max_pt
        self.jet_min_eta = config.jet_min_eta
        self.jet_max_eta = config.jet_max_eta

        # filter on topocluster
        self.topo_filter = config.topo_filter
        self.topo_min_pt = config.topo_min_pt
        self.topo_max_pt = config.topo_max_pt
        self.topo_min_eta = config.topo_min_eta
        self.topo_max_eta = config.topo_max_eta

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.files)

        # one file = one event = multiple clusters
        for i, file in enumerate(self.files):
            # test if end of iteration
            if i > self.max_iter:
                break
            # list of clusters
            clusters = pickle_load(self.path + "/" + file, verbose=False)
            if self.shuffle_clusters:
                random.shuffle(clusters)

            for cluster in clusters:
                jet = (cluster["jet_pt"], cluster["jet_eta"], cluster["jet_phi"])
                if self.jet_filter:
                    if jet == (0, 0, 0):
                        continue
                    if not ((self.jet_min_pt < jet[0] < self.jet_max_pt) and
                        (self.jet_min_eta < jet[1] < self.jet_max_eta)):
                        continue
                    
                if self.topo_filter:
                    if not ((self.topo_min_pt < cluster["topo_pt"] < self.topo_max_pt) and
                        (self.topo_min_eta < abs(cluster["topo_eta"]) < self.topo_max_eta)):
                        continue

                if self.preprocess is not None:
                    cluster = self.preprocess(cluster)

                yield cluster, i
