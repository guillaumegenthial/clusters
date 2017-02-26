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


## Run

- for simple features, with `config` being the name of the config file in `/configs/`

```
python flat.py -c config
```

- for np array input

```
python square.py -c fc_simple
```