# SLAC


- connect to `ssh username@atlint03.slac.stanford.edu`
- change shell to bash
- edit .bash_profile and add. the `root/proof/` is too old and create some problems!!
```
export ROOTSYS=/afs/slac.stanford.edu/g/atlas/packages/root/root
export XPDUTILS=$ROOTSYS/etc/proof/utils
export PATH=${ROOTSYS}/bin:${PATH}
export LD_LIBRARY_PATH=${PYTHIA8}/lib:${ROOTSYS}/lib:${ROOTSYS}/bin/root:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=${ROOTSYS}/lib:${ROOTSYS}/bin/root:${DYLD_LIBRARY_PATH}
export PYTHONPATH=$ROOTSYS/lib:${PYTHONPATH}
```

- use anaconda with `anaconda`
- create env (in that case `clustersenv`) with `conda create -n clustersenv python=2.7 anaconda`
- activate env with `source activate clustersenv`
- install new package with `conda install -n yourenvname [package]`