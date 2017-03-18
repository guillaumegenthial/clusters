class general(object):
    # general data
    path = "data/events2"
    max_iter = 46
    train_files = "data/config_{}/train.txt".format(max_iter)
    dev_files = "data/config_{}/dev.txt".format(max_iter)
    test_files = "data/config_{}/test.txt".format(max_iter)
    shuffle = False
    dev_size = 0.1
    test_size = 0.2
    config_files = __file__.split(".")[-2] + ".py"

    # prop data
    jet_filter = False 
    jet_min_pt = 20 
    jet_max_pt = 2000 
    jet_min_eta = 0 
    jet_max_eta = 1  
    topo_filter = False 
    topo_min_pt = -10000
    topo_max_pt = 10000
    topo_min_eta = 0 
    topo_max_eta = 1
    shuffle_clusters = True
    part_filter = True
    part_min = 1

    # pred
    output_size = 2
    baseclass = 0

    # training
    batch_size = 20
    n_epochs = 10
    restore = False
    dropout = 0.5
    reg = 0.001
    lr = 0.001
    selection = "f1" or "acc"
    f1_mode = "macro"
    early_stopping = True
    nb_ep_no_imprvmt = 2
    max_training_time = 10
