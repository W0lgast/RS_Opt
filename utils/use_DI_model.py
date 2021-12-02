"""

Runs training for deepInsight

"""
# -----------------------------------------------------------------------

from deep_insight.options import get_opts, get_globals
from deep_insight.wavelet_dataset import create_train_and_test_datasets, WaveletDataset
from deep_insight.trainer import Trainer
import deep_insight.loss
import deep_insight.networks
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

GLOBALS = get_globals()

hdf5_file = h5py.File(GLOBALS.h5_path, mode='r')
wavelets = np.array(hdf5_file['inputs/wavelets'])
loss_functions = GLOBALS.loss_functions

loss_weights = GLOBALS.loss_weights

# ..todo: second param is unneccecary at this stage, use two empty arrays to match signature but it doesn't matter
training_options = get_opts(GLOBALS.h5_path, train_test_times=(np.array([]), np.array([])))
training_options['loss_functions'] = loss_functions.copy()
training_options['loss_weights'] = loss_weights
training_options['loss_names'] = list(loss_functions.keys())
training_options['shuffle'] = False

exp_indices = np.arange(0, wavelets.shape[0] - training_options['model_timesteps'])
cv_splits = np.array_split(exp_indices, training_options['num_cvs'])

training_indices = []
for arr in cv_splits[0:-1]:
    training_indices += list(arr)
training_indices = np.array(training_indices)

test_indeces = np.array(cv_splits[-1])
# opts -> generators -> model
# reset options for this cross validation set
training_options = get_opts(GLOBALS.h5_path, train_test_times=(training_indices, test_indeces))
training_options['loss_functions'] = loss_functions.copy()
training_options['loss_weights'] = loss_weights
training_options['loss_names'] = list(loss_functions.keys())
training_options['shuffle'] = False
training_options['random_batches'] = False

train_dataset, test_dataset = create_train_and_test_datasets(training_options, hdf5_file)
model_function = getattr(deep_insight.networks, train_dataset.model_function)
MODEL = model_function(train_dataset)
MODEL.load_state_dict(torch.load(GLOBALS.model_path))
MODEL.eval()

def get_odometry(data, get_pos=False, get_ffc=False):
    tansor = torch.from_numpy(data).unsqueeze(0)
    logits = MODEL(tansor, False, get_ffc)
    position_ests = list(logits[GLOBALS.targets.index("position")])[0]
    angle_ests = list(logits[GLOBALS.targets.index("direction")])[0]
    speed_ests = list(logits[GLOBALS.targets.index("speed")])[0]
    if get_pos:
        if get_ffc == False:
            return speed_ests[0].item(), angle_ests.item(), position_ests.detach().numpy() #+pi for felix
        else:
            ffc = logits[-1]
            return speed_ests[0].item(), angle_ests.item(), position_ests.detach().numpy(), ffc
    else:
        if get_ffc == False:
            return speed_ests[0].item(), angle_ests.item()#-np.pi/2
        else:
            ffc = logits[-1]
            return speed_ests[0].item(), angle_ests.item(), ffc#-np.pi/2