import torch
from mdu.randomness import set_all_seeds
import numpy as np
from mdu.nn.load_models import get_model
from mdu.nn.constants import ModelName
from mdu.optim.train import train_ensembles
import torch.nn as nn
from mdu.vis.toy_plots import plot_decision_boundaries, plot_uncertainty_measures
from mdu.unc.constants import UncertaintyType
from mdu.unc.risk_metrics.constants import GName, RiskType, ApproximationType
from mdu.unc.multidimensional_uncertainty import MultiDimensionalUncertainty
from mdu.eval.eval_utils import get_ensemble_predictions


torch.manual_seed(0)
np.random.seed(0)

set_all_seeds(42)

from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split


