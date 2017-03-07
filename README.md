# ATLAS Calorimeter Cluster Splitting

## Modules
ROOT / Tensorflow


## Task

- Predict number of particle in ATLAS topo cluster


## Data

- 2k Monte-Carlo simulated events in a ROOT tree
- 600k extracted clusters in jets > 20 GeV and eta < 0.5
- input data:
    - simple features
    - fill np arrays with energy deposit

## Models

- Fully connected network on simple features
- Fully connected network on flattened np arrays
- Convolutional layers on np arrays
- rotation invariant extraction from embedding of cells

## Test

Add flag -t

```
python simple.py -t
``` 

## Run

- for simple features, with `config` being the name of the config file in `/configs/`

```
python simple.py -c config
```

- for layers features (stack of arrays for each layer)

```
python layers.py -c config
```

- for embeddings
```
python embeddings.py -c config
```