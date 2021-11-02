"""

Defines options dict for training.

"""

import deep_insight.loss

RAT_NAME = "POSTSKIP"
RAT_NAME = "Felix"

MODEL_PATH = f"models/{RAT_NAME}.pt"
MAT_PATH = f"data/{RAT_NAME}.mat"
H5_PATH = f"data/{RAT_NAME}.h5"

TARGETS = ["position", "head_direction", "direction", "speed"]

loss_functions = {'position': 'euclidean_loss',
                  'head_direction': 'cyclical_mae_rad',
                  'direction': 'cyclical_mae_rad',
                  'direction_delta': 'cyclical_mae_rad',
                  'speed': 'mae'}
LOSS_FUNCTIONS = {t: loss_functions[t] for t in TARGETS}

loss_weights = {'position': 1,
                'head_direction': 25,
                'direction': 25,
                'direction_delta': 25,
                'speed': 2}
LOSS_WEIGHTS = {t: loss_weights[t] for t in TARGETS}



def get_opts(fp_hdf_out, train_test_times):
    """
    Returns the options dictionary which contains all parameters needed to train the model

    ..todo:: which of these opts are redundant? take them out.
    """
    opts = dict()

    # -------- DATA ------------------------
    opts['fp_hdf_out'] = fp_hdf_out  # Filepath for hdf5 file storing wavelets and outputs
    opts['sampling_rate'] = 512*4 # Sampling rate of the wavelets
    opts['training_indices'] = train_test_times[0].tolist()  # Indices into wavelets used for training the model, adjusted during CV
    opts['testing_indices'] = train_test_times[1].tolist()  # Indices into wavelets used for testing the model, adjusted during CV
    #opts['channels'] = 16
    opts['channels'] = None

    # -------- MODEL PARAMETERS --------------
    opts['model_function'] = 'Standard_Decoder'  # Model architecture used
    opts['model_timesteps'] = 64
    # 32 #64  # How many timesteps are used in the input layer, e.g. a sampling rate of 30 will yield 2.13s windows. Has to be divisible X times by 2. X='num_convs_tsr'
    opts['num_convs_tsr'] = 5  # Number of downsampling steps within the model, e.g. with model_timesteps=64, it will downsample 64->32->16->8->4 and output 4 timesteps
    opts['average_output'] = 2**opts['num_convs_tsr']  # Whats the ratio between input and output shape

    opts['optimizer'] = 'adam'  # Learning algorithm
    opts['learning_rate'] = 0.0007  # Learning rate
    opts['kernel_size'] = 3  # Kernel size for all convolutional layers
    opts['conv_padding'] = 'same'  # Which padding should be used for the convolutional layers
    opts['act_conv'] = 'ELU'  # Activation function for convolutional layers
    opts['act_fc'] = 'ELU'  # Activation function for fully connected layers
    opts['dropout_ratio'] = 0  # Dropout ratio for fully connected layers
    opts['filter_size'] = 64  # Number of filters in convolutional layers
    opts['num_units_dense'] = 1024  # Number of units in fully connected layer
    opts['num_dense'] = 2  # Number of fully connected layers
    opts['gaussian_noise'] = 1  # How much gaussian noise is added (unit = standard deviation)

    # -------- TRAINING----------------------
    opts['batch_size'] = 8  # Batch size used for training the model
    opts['steps_per_epoch'] = 250  # Number of steps per training epoch
    opts['validation_steps'] = 15  # Number of steps per validation epoch #..todo: val happens once per epoch now, this var is redundant
    opts['epochs'] = 250  # Number of epochs
    opts['shuffle'] = False  # If input should be shuffled
    opts['random_batches'] = True  # If random batches in time are used
    opts['num_cvs'] = 3 # the number of cross validation splits

    # -------- MISC--------------- ------------
    opts['tensorboard_logfolder'] = './'  # Logfolder for tensorboard
    opts['model_folder'] = './'  # Folder for saving the model
    opts['log_output'] = False  # If output should be logged
    opts['save_model'] = False  # If model should be saved

    return opts


# def get_opts_old(fp_hdf_out, train_test_times):
#     """
#     Returns the options dictionary which contains all parameters needed to train the model
#
#     ..todo:: which of these opts are redundant? take them out.
#     """
#     opts = dict()
#
#     # -------- DATA ------------------------
#     opts['fp_hdf_out'] = fp_hdf_out  # Filepath for hdf5 file storing wavelets and outputs
#     opts['sampling_rate'] = 30  # Sampling rate of the wavelets
#     opts['training_indices'] = train_test_times[0].tolist()  # Indices into wavelets used for training the model, adjusted during CV
#     opts['testing_indices'] = train_test_times[1].tolist()  # Indices into wavelets used for testing the model, adjusted during CV
#
#     # -------- MODEL PARAMETERS --------------
#     opts['model_function'] = 'Standard_Decoder'  # Model architecture used
#     opts['model_timesteps'] = 64  # How many timesteps are used in the input layer, e.g. a sampling rate of 30 will yield 2.13s windows. Has to be divisible X times by 2. X='num_convs_tsr'
#     opts['num_convs_tsr'] = 4  # Number of downsampling steps within the model, e.g. with model_timesteps=64, it will downsample 64->32->16->8->4 and output 4 timesteps
#     opts['average_output'] = 2**opts['num_convs_tsr']  # Whats the ratio between input and output shape
#
#     opts['optimizer'] = 'adam'  # Learning algorithm
#     opts['learning_rate'] = 0.0007  # Learning rate
#     opts['kernel_size'] = 3  # Kernel size for all convolutional layers
#     opts['conv_padding'] = 'same'  # Which padding should be used for the convolutional layers
#     opts['act_conv'] = 'ELU'  # Activation function for convolutional layers
#     opts['act_fc'] = 'ELU'  # Activation function for fully connected layers
#     opts['dropout_ratio'] = 0  # Dropout ratio for fully connected layers
#     opts['filter_size'] = 64  # Number of filters in convolutional layers
#     opts['num_units_dense'] = 1024  # Number of units in fully connected layer
#     opts['num_dense'] = 2  # Number of fully connected layers
#     opts['gaussian_noise'] = 1  # How much gaussian noise is added (unit = standard deviation)
#
#     # -------- TRAINING----------------------
#     opts['batch_size'] = 8  # Batch size used for training the model
#     opts['steps_per_epoch'] = 250  # Number of steps per training epoch
#     opts['validation_steps'] = 250  # Number of steps per validation epoch #..todo: val happens once per epoch now, this var is redundant
#     opts['epochs'] = 15  # Number of epochs
#     opts['shuffle'] = True  # If input should be shuffled
#     opts['random_batches'] = True  # If random batches in time are used
#     opts['num_cvs'] = 5 # the number of cross validation splits
#
#     # -------- MISC--------------- ------------
#     opts['tensorboard_logfolder'] = './'  # Logfolder for tensorboard
#     opts['model_folder'] = './'  # Folder for saving the model
#     opts['log_output'] = False  # If output should be logged
#     opts['save_model'] = False  # If model should be saved
#
#     return opts
