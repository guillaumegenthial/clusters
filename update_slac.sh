#!/bin/bash
rsync -rtv --exclude-from='.rsyncignore' . genthial@atlint03.slac.stanford.edu:~/private/clusters
