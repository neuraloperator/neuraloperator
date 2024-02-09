"""
Training a UQNO on Darcy-Flow
=============================

In this example, we demonstrate how to train a UQNO for uncertainty quantification for a given base model on a Darcy flow problem

"""

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO, UQNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from utils import *
import numpy as np

device = 'cuda'
quantile_model_config = {
    "n_modes": (6,6),
    "hidden_channels": 12,
    "projection_channels": 12,
    "learning_rate": 0.001,
    "epochs": 100,
    "weight_decay": 3e-4,
    "batch_size": 2,
    "T_max": 200}
main_model_config = quantile_model_config # for simplicity, assume main model has the same config

# load trained base model
base_model = load_fno_model("https://drive.google.com/uc?export=download&id=1xGy1rvmR4w3_bwWGfiA9A3v6nrb5fEq9", "main_model", main_model_config)
quantile_model = load_fno_model("https://drive.google.com/uc?export=download&id=1G9NX2TiyNhkoSHmw8wK3yVbtZsGVeNYZ", "quantile_model", quantile_model_config)

# load data, note this training model is reserved for UQNO, and it should be
# separate from the training set of base model
main_y_encoder = load_base_model_encoder()
download_darcy421_data("darcy_x.mat", "darcy_y.mat")
calib_loader, test_loader = get_calib_test_loaders("darcy_x.mat", "darcy_y.mat", quantile_model_config["batch_size"])

# initialize a (0.05, 0.10) UQNO
alpha = 0.05 
delta = 0.05
uqno = UQNO(alpha, delta, base_model, main_y_encoder, quantile_model_config)

# get residual of base model
#x_train, residual_train = uqno.get_residual(main_model, main_y_encoder, train_loader)
x_val, residual_val = uqno.get_residual(base_model, main_y_encoder, calib_loader)
x_test, residual_test = uqno.get_residual(base_model, main_y_encoder, test_loader)
residual_encoder = load_residual_model_encoder()
calib_residual_loader, _ = get_darcy_loader_data(x_val, residual_val, quantile_model_config["batch_size"], positional_encoding=False, shuffle=False, encode_output=False)

# train calibrated uqno
n_calib_samples = x_val.shape[0]
discretization = residual_val.view(n_calib_samples, -1).shape[1]

# for demo purposes, we supply a quantile model checkpoint rather than training from scratch
# when training from scratch, set use_pretrained_quantile_model to False (which is the default)
train_residual_loader = None # put actual train loader here, None since we are taking a trained model
uqno.quantile_model = quantile_model
uqno.train_calibrated_uqno(train_residual_loader, calib_residual_loader, residual_encoder, n_calib_samples, discretization, use_pretrained_quantile_model=True)

# evaluate
# predict_with_uncertainty returns full batch results, works if sample size is small
# point_pred, uq_pred = uqno.predict_with_uncertainty(test_loader)
# for large samples (where the full batch cannot fit in memory), eval_coverage_bandwidth
# evaluates coverage and bandwidth in a streaming manner
# note that our method is conservative when the size of validation set is small
calibrated_percentage, avg_bandwidth = uqno.eval_coverage_bandwidth(test_loader)