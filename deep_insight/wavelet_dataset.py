from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math
from utils.misc import getspeed

def create_train_and_test_datasets(opts, hdf5_file, train_half=None):
    """
    Creates training and test datasets from given opts dictionary and hdf5 file.
    """
    # 1.) Create training generator
    training_generator = WaveletDataset(opts, hdf5_file, training=True, train_half=train_half)
    # 2.) Create testing generator
    if train_half is not None:
        if train_half == "top": train_half_n = "bottom"
        elif train_half == "bottom": train_half_n = "top"
        train_half = train_half_n
    testing_generator = WaveletDataset(opts, hdf5_file, training=False, train_half=train_half)

    return training_generator, testing_generator

class WaveletDataset(Dataset):
    """
    Dataset containing raw wavelet sequence; __getitem__ returns an (input, output) pair.
    ..todo: better docs
    """
    def __init__(self, opts, hdf5_file, training, train_half=None):
        # 1.) Set all options as attributes
        self.set_opts_as_attribute(opts)
        # 2.) Load data memmaped for mean/std estimation and fast plotting
        self.wavelets = np.array(hdf5_file['inputs/wavelets'])

        self.last_pos = None
        self.last_speed = None
        self.last_ang = None
        self.prev_ind = None

        if (train_half is not None) and (train_half in ["top", "bottom"] == False):
            print("ERROR: unknown train half keyword")
            exit(0)

        # train half will mean training will only occur on top half of maze
        self.train_half = train_half

        # Get output(s)
        outputs = []
        # the loss function dict has a key representing each output - loop through them and
        # get the outputs
        for key, value in opts['loss_functions'].items():
            tmp_out = hdf5_file['outputs/' + key]
            outputs.append(tmp_out)
        self.outputs = [np.array(o) for o in outputs]

        # 3.) Prepare for training
        self.training = training
        self.prepare_data_generator(training=training)

    def __len__(self):
        return len(self.cv_indices)

    def __getitem__(self, idx):
        # 1.) Define start and end index
        og_idx = idx
        if self.shuffle:
            idx = np.random.choice(self.cv_indices)
        else:
            idx = self.cv_indices[idx]
        cut_range = np.arange(idx, idx + self.sample_size)
        past_cut_range = np.arange(idx-1,  idx+self.sample_size-1)
        # 2.) Above takes consecutive batches, below takes random batches
        if self.random_batches:
            start_index = np.random.choice(self.cv_indices, size=1)[0]
            cut_range = np.arange(start_index, start_index + self.model_timesteps)
            past_cut_range = np.arange(start_index-1, start_index+self.model_timesteps-1)

        # 3.) Get output sample
        output_sample = self.get_output_sample(cut_range, past_cut_range)
        # if train_half, make sure point is from top half of maze
        if output_sample[0][1] > 150 and self.train_half == "bottom":
            return self.__getitem__(og_idx)
        if output_sample[0][1] <= 150 and self.train_half == "top":
            return self.__getitem__(og_idx)

        # 4.) Get input sample
        input_sample = self.get_input_sample(cut_range)

        self.prev_ind = idx
        #print(idx)

        return (input_sample, output_sample)

    # -------------------------------------------------------------------------
    # Public Function
    # -------------------------------------------------------------------------

    def prepare_data_generator(self, training):
        # Define sample size and means
        self.sample_size = self.model_timesteps
        # define indices depending on whether training or testing
        if training:
            self.cv_indices = self.training_indices
        else:
            self.cv_indices = self.testing_indices
        self.est_mean = np.median(self.wavelets[self.training_indices, :, :], axis=0)
        self.est_std = np.median(abs(self.wavelets[self.training_indices, :, :] - self.est_mean), axis=0)
        # Define output shape. Most robust way is to get a dummy input and take that shape as output shape
        (dummy_input, dummy_output) = self.__getitem__(0)
        # Corresponds to the output of this generator, aka input to model. Also remove batch shape,
        self.input_shape = dummy_input.shape[:]

    def set_opts_as_attribute(self, opts):
        """
        Sets all entries in opts dict to internal parameters with names equal to the
        option keys.
        """
        for k, v in opts.items():
            setattr(self, k, v)

    def get_input_sample(self, cut_range):
        # 1.) Cut Ephys / fancy indexing for memmap is planned, if fixed use: cut_data = self.wavelets[cut_range, self.fourier_frequencies, self.channels]
        cut_data = self.wavelets[cut_range, :, :]

        # 2.) Normalize input
        cut_data = (cut_data - self.est_mean) / self.est_std

        # 3.) Reshape for model input
        #cut_data = np.reshape(cut_data, (self.batch_size, self.model_timesteps, cut_data.shape[1], cut_data.shape[2]))

        # 4.) Take care of optional settings
        cut_data = np.transpose(cut_data, axes=(2, 0, 1))
        cut_data = cut_data[..., np.newaxis]

        return cut_data

    def get_output_sample(self, cut_range, prev_cut_range):
        # 1.) Cut Ephys
        out_sample = []
        for i, out in enumerate(self.outputs):

            var_name = list(self.loss_functions.keys())[i]
            cut_data = out[cut_range, ...]
            pcd = out[prev_cut_range, ...]

            # 3.) Divide evenly and make sure last output is being decoded

            # if self.average_output:
            #     cut_data = cut_data[np.arange(0, cut_data.shape[0] + 1, self.average_output)[1::] - 1]
            # out_sample.append(cut_data)
            ## ..todo: replaced above with below: kipp!

            if var_name == 'position':
                pcdm = np.mean(pcd, axis=0)
                cut_data_m = np.mean(cut_data, axis=0)
                #cut_data_m = cut_data[-1, :]
                #out_sample.append(cut_data_m)
                #pcdm = pcd[-1,:]
                # ..todo: below is average, above is final

                out_sample.append(cut_data_m)

            elif var_name == 'head_direction':
                #out_sample.append(cut_data[-1][0])
                # ..todo: this is taking the average head direction for a window - above takes final head direction
                dirr = np.mean([c[0] for c in cut_data])
                print(f"head direction = {dirr}")
                out_sample.append(dirr)
            elif var_name == 'direction':
                # ch_y = bookends[1][1] - bookends[0][1]
                # ch_x = bookends[1][0] - bookends[0][0]
                # ang_rad = math.atan2(ch_x, ch_y)
                # out_sample.append(ang_rad*(180/np.pi))
                # return angle diff between cut data m and last pos
                dirt = math.atan2(cut_data_m[1]-pcdm[1], cut_data_m[0]-pcdm[0])

                #dirt = np.mean([c[0] for c in cut_data])
                out_sample.append(dirt)
                #print(f"travel direction = {dirt}")
            elif var_name == 'speed':
                spd = getspeed(cut_data_m, pcdm)
                out_sample.append(spd)
            else:
                print("ERROR: Unknown var name!")
                exit(0)

        return out_sample

class WaveletDatasetFrey(Dataset):
    """
    Dataset containing raw wavelet sequence; __getitem__ returns an (input, output) pair.
    ..todo: better docs
    """
    def __init__(self, opts, hdf5_file, training):
        # 1.) Set all options as attributes
        self.set_opts_as_attribute(opts)
        # 2.) Load data memmaped for mean/std estimation and fast plotting
        self.wavelets = np.array(hdf5_file['inputs/wavelets'])

        # Get output(s)
        outputs = []
        # the loss function dict has a key representing each output - loop through them and
        # get the outputs
        for key, value in opts['loss_functions'].items():
            tmp_out = hdf5_file['outputs/' + key]
            outputs.append(tmp_out)
        self.outputs = [np.array(o) for o in outputs]

        # 3.) Prepare for training
        self.training = training
        self.prepare_data_generator(training=training)

    def __len__(self):
        return len(self.cv_indices)

    def __getitem__(self, idx):
        # 1.) Define start and end index
        if self.shuffle:
            idx = np.random.choice(self.cv_indices)
        else:
            idx = self.cv_indices[idx]
        cut_range = np.arange(idx, idx + self.sample_size)
        # 2.) Above takes consecutive batches, below takes random batches
        if self.random_batches:
            start_index = np.random.choice(self.cv_indices, size=1)[0]
            cut_range = np.arange(start_index, start_index + self.model_timesteps)

        # 3.) Get input sample
        input_sample = self.get_input_sample(cut_range)

        # 4.) Get output sample
        output_sample = self.get_output_sample(cut_range)

        return (input_sample, output_sample)

    # -------------------------------------------------------------------------
    # Public Function
    # -------------------------------------------------------------------------

    def prepare_data_generator(self, training):
        # Define sample size and means
        self.sample_size = self.model_timesteps
        # define indices depending on whether training or testing
        if training:
            self.cv_indices = self.training_indices
        else:
            self.cv_indices = self.testing_indices
        self.est_mean = np.median(self.wavelets[self.training_indices, :, :], axis=0)
        self.est_std = np.median(abs(self.wavelets[self.training_indices, :, :] - self.est_mean), axis=0)
        # Define output shape. Most robust way is to get a dummy input and take that shape as output shape
        (dummy_input, dummy_output) = self.__getitem__(0)
        # Corresponds to the output of this generator, aka input to model. Also remove batch shape,
        self.input_shape = dummy_input.shape[:]

    def set_opts_as_attribute(self, opts):
        """
        Sets all entries in opts dict to internal parameters with names equal to the
        option keys.
        """
        for k, v in opts.items():
            setattr(self, k, v)

    def get_input_sample(self, cut_range):
        # 1.) Cut Ephys / fancy indexing for memmap is planned, if fixed use: cut_data = self.wavelets[cut_range, self.fourier_frequencies, self.channels]
        cut_data = self.wavelets[cut_range, :, :]

        # 2.) Normalize input
        cut_data = (cut_data - self.est_mean) / self.est_std

        # 3.) Reshape for model input
        #cut_data = np.reshape(cut_data, (self.batch_size, self.model_timesteps, cut_data.shape[1], cut_data.shape[2]))

        # 4.) Take care of optional settings
        cut_data = np.transpose(cut_data, axes=(2, 0, 1))
        cut_data = cut_data[..., np.newaxis]

        return cut_data

    def get_output_sample(self, cut_range):
        # 1.) Cut Ephys
        out_sample = []
        for out in self.outputs:
            cut_data = out[cut_range, ...]

            # 3.) Divide evenly and make sure last output is being decoded
            if self.average_output:
                cut_data = cut_data[np.arange(0, cut_data.shape[0] + 1, self.average_output)[1::] - 1]
            out_sample.append(cut_data)

        return out_sample