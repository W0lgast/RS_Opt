"""

Runs training for deepInsight

"""
# -----------------------------------------------------------------------

from deep_insight.options import get_opts, RAT_NAME, MODEL_PATH, H5_PATH, LOSS_FUNCTIONS, LOSS_WEIGHTS, TRAIN_HALF_KEY
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
import sys
sys.setrecursionlimit(10000)

# -----------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
USE_WANDB = True

if __name__ == '__main__':

    # for rat_name in ["Elliott", "Felix", "Gerritt", "Herman", "Ibsen", "Jasper"]:

    if USE_WANDB: wandb.init(project=RAT_NAME)

    PREPROCESSED_HDF5_PATH = H5_PATH
    hdf5_file = h5py.File(PREPROCESSED_HDF5_PATH, mode='r')
    wavelets = np.array(hdf5_file['inputs/wavelets'])
    frequencies = np.array(hdf5_file['inputs/fourier_frequencies'])

    loss_functions = LOSS_FUNCTIONS
    loss_weights = LOSS_WEIGHTS

    training_options = get_opts(PREPROCESSED_HDF5_PATH, train_test_times=(np.array([]), np.array([])))

    exp_indices = np.arange(0, wavelets.shape[0] - training_options['model_timesteps'])
    cv_splits = np.array_split(exp_indices, training_options['num_cvs'])

    for cv_run, cvs in enumerate(cv_splits):
        # For cv
        training_indices = np.setdiff1d(exp_indices, cvs)  # All except the test indices
        testing_indices = cvs
        # opts -> generators -> model
        # reset options for this cross validation set
        training_options = get_opts(PREPROCESSED_HDF5_PATH, train_test_times=(training_indices, testing_indices))
        training_options['loss_functions'] = loss_functions.copy()
        training_options['loss_weights'] = loss_weights
        training_options['loss_names'] = list(loss_functions.keys())

        train_dataset, test_dataset = create_train_and_test_datasets(training_options, hdf5_file,
                                                                     train_half=TRAIN_HALF_KEY)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=training_options['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=training_options['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True)

        model_function = getattr(deep_insight.networks, train_dataset.model_function)
        model = model_function(train_dataset)

        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=train_dataset.learning_rate,
                                     amsgrad=True)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=(loss_functions, loss_weights),
            optimizer=optimizer,
            device=DEVICE,
            use_wandb=USE_WANDB
        )

        trainer.train()

        torch.save(model.state_dict(), MODEL_PATH)
        print("Done!")
        exit(0)
