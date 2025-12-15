# FEDERATED UNCERTAINTY

### Instruction

1. Train your ensembles, using the script

```
./federated_uncertainty_scripts/run_main_federated_ensembles.sh
```

2. Evaluate uncertainty metrics via

```
./federated_uncertainty_scripts/run_compute_measures_ood.sh
```

### TinyImageNet Installation

If you want to run experiments on tiny-imagenet dataset, you should complete following steps:

1. Install the dataset to the path
```
FEDERATED-UNCERTAINTY/data/tiny-imagenet-200
```
For example, you may get it [here](https://cs231n.stanford.edu/tiny-imagenet-200.zip)

2. Preprocess data
```
python federated_uncertainty/data/imagenet_parser.py 
```
