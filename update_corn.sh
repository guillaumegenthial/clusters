#!/bin/bash
rsync -rtv --exclude 'data/ntuple_v3_2000k.npy' . genthial@corn.stanford.edu:~/private/clusters